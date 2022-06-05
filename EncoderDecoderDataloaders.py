# Keep alphabetically ordered to minimize duplicate imports
from conllu import parse
from conllu import parse_incr
import csv
from datasets import load_dataset
import datetime
from functools import partial
from datasets import load_metric
import gc
import matplotlib.pyplot as plt
from torch.optim import AdamW
import numpy as np
import os
import pandas as pd
from plotly.tools import FigureFactory as ff
import seaborn as sn

import transformers
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, BertForTokenClassification, AutoTokenizer, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm as std_tqdm
tqdm =partial(std_tqdm, leave=False, position=0, dynamic_ncols=True)
from typing import *
import warnings
import zipfile
import nltk
from nltk.corpus import wordnet
from dataloading_utils import filter_negative_hundred, TransformerCompatDataset, get_num_examples, get_validation_acc, get_dataset_mapping
from ArkDataset.load_ark import load_ark
from TPANNDataset.load_tpann import load_tpann
from TweeBankDataset.load_tweebank import load_tweebank

ark_train, ark_val, ark_test = load_ark()
tpann_train, tpann_val, tpann_test = load_tpann()
tweebank_train, tweebank_val, tweebank_test = load_tweebank()
ark_all = ark_train + ark_val + ark_test
tweebank_all = tweebank_train + tweebank_val + tweebank_test
# Needed Imports
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
def get_shared_examples(ark_all,twee_all):
  """Takes as input all examples from ark and twee bank, then returns a list of shared examples of the format
  [shared_x,ark_label,twee_label]"""
  shared_examples = []
  for ark in tqdm(ark_all):
    for twee in tweebank_all:
      if ark[0]==twee[0]:
        shared_examples.append([])
        shared_examples[-1].extend([ark[0],ark[1],twee[1]]) 
  print("Number of shared Examples: ", len(shared_examples))
  return shared_examples
shared_examples = get_shared_examples(ark_all,tweebank_all)


def get_dataloader(model_name:str, dataset, batch_size, shuffle=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, use_fast=True, model_max_length=512)
    if model_name == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compat_dataset = TransformerCompatDataset(dataset, tokenizer)
    dataloader = DataLoader(compat_dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=data_collator)
    return dataloader
class ArkSharedDataset(torch.utils.data.Dataset):
    def __init__(self, shared_examples,test=False):
        self.shared_examples = shared_examples
        self.X = []
        self.Y = []
        for ex in shared_examples:
          self.X.append(ex[0])
          self.Y.append(ex[1])
        if test:
          self.X = self.X[-50:]
          self.Y = self.Y[-50:]
        assert len(self.X)==len(self.Y)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        x = self.X[ind]
        y = self.Y[ind]
        return x, y

class TweeSharedDataset(torch.utils.data.Dataset):
    def __init__(self, shared_examples,test=False):
        self.shared_examples = shared_examples
        self.X = []
        self.Y = []
        for ex in shared_examples:
          self.X.append(ex[0])
          self.Y.append(ex[2])
        if test:
          self.X = self.X[-50:]
          self.Y = self.Y[-50:]
        assert len(self.X)==len(self.Y)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        x = self.X[ind]
        y = self.Y[ind]
        return x, y
    

ark_shared = ArkSharedDataset(shared_examples,test=False)
twee_shared = TweeSharedDataset(shared_examples,test=False)


ark_shared_dataloader = get_dataloader("gpt2",ark_shared,20,shuffle=False )
twee_shared_dataloader = get_dataloader("gpt2",twee_shared,20,shuffle=False )

# print(len(ark_shared))
# print(len(twee_shared))


class SharedDataLoader:
    def __init__(self,shared_examples,batch_size):
        self.ark_shared_data = ArkSharedDataset(shared_examples,test=False)
        self.twee_shared_data = TweeSharedDataset(shared_examples,test=False)

        self.ark_loader =get_dataloader("gpt2",self.ark_shared,batch_size,shuffle=False )
        self.twee_loader =get_dataloader("gpt2",self.twee_shared,batch_size,shuffle=False )
        assert len(self.ark_shared_data) == len(self.twee_shared_data)
        self.length = len(self.ark_shared_data)
    def __iter__(self):
        for dataArk,dataTwee in zip(self.ark_loader,self.twee_loader):
          batch = dataArk
          batch['twee_labels'] = dataTwee['labels']
          x = {}
          for key in batch:
            if key !="labels" and key!="twee_labels":
              x[key] = batch[key]
            elif key =="labels":
              y = batch[key]
            elif key == "twee_labels":
              z = batch[key]
          yield x,y,z
SharedLoader = SharedDataLoader(shared_examples,batch_size=20)
for x,y,z in SharedLoader:
  print(x)
  print(y)
  print(z)
  break