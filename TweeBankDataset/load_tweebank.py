import torch
import torch.utils.data
import numpy as np
from typing import *
from conllu import parse_incr
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_reader(data_path):
    """  Some useful info     
            https://rdrr.io/cran/NLP/man/CoNLLUTextDocument.html
            LEMMA (lemma or stem of word form), 
            UPOSTAG (universal part-of-speech tag, see https://universaldependencies.org/u/pos/index.html), 
            XPOSTAG (language-specific part-of-speech tag, may be 
    """
    with open(data_path, "r", encoding="utf-8") as data_file:
        X = []
        Y = []
        for tokenlist in parse_incr(data_file):
            X.append([])
            Y.append([])
            for token in tokenlist:
                X[-1].append(str(token))
                Y[-1].append(token['upos'])
    return X, Y

def make_pos_mapping(data_paths: List[str]):
    all_pos = set()
    def update_pos(tokenlist):
        for elem in tokenlist:
            all_pos.add(elem['upos'])
    
    def tokenz(data_path):
        with open(data_path, "r", encoding="utf-8") as data_file:
            for tokenlist in parse_incr(data_file):
                update_pos(tokenlist)
    for path in data_paths:
        tokenz(path)
    return dict(zip(list(all_pos),np.arange(len(list(all_pos)))))

twee_train_path = "TweeBankDataset/Tweebank-dev/en-ud-tweet-train.conllu"
twee_dev_path = "TweeBankDataset/Tweebank-dev/en-ud-tweet-dev.conllu"
twee_test_path = "TweeBankDataset/Tweebank-dev/en-ud-tweet-test.conllu"
TWEEBANK_POS_MAPPING = make_pos_mapping([twee_train_path, twee_dev_path, twee_test_path])

# Create Dataset Classes
class TweebankTrain(torch.utils.data.Dataset):
    def __init__(self, data_path): 
        self.Data_dir = data_path 
        self.X, self.Yraw = data_reader(data_path)
        self.Y = []
        self.num_labels = 17
        for ex in self.Yraw:
          self.Y.append([])
          for elem in ex:
            self.Y[-1].append(TWEEBANK_POS_MAPPING[elem])
          
        assert(len(self.X) == len(self.Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        X = self.X[ind]
        Y = self.Y[ind]
        Y = torch.as_tensor(Y, dtype=torch.long, device=device) 
        return X, Y
   
def load_tweebank():
    tweebank_train = TweebankTrain(twee_train_path)
    tweebank_val = TweebankTrain(twee_dev_path)
    tweebank_test = TweebankTrain(twee_test_path)
    return tweebank_train, tweebank_val, tweebank_test