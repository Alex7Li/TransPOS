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



# Needed Imports
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

with open("ark_twee_shared.pkl",'rb') as f:
  shared_examples = pickle.load(f)


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

print(len(ark_shared))
print(len(twee_shared))

for x,xx in zip(ark_shared_dataloader,twee_shared_dataloader):
  batch = x
  batch['twee_labels'] = xx['labels']
  print(batch)
  break