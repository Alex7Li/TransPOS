from typing import List
import torch
import torch.utils.data
import numpy as np
from conllu import parse_incr

device = "cuda" if torch.cuda.is_available() else "cpu"


# Atis doesn't actually use PUNCT, SCONJ, SYM, or X
# It doesn't have SYM PUNCT, or X in the dataset originally
# because the data is clean
# and SCONJ is also super rare - a bertweet with 94% accuracy
# detected only 32
# # cases of it out of something like 50k tokens in the validation set.
# But we use this extended set, since it's equivalent to tweebank.
ATIS_POS_MAPPING = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "AUX": 3,
    "CCONJ": 4,
    "DET": 5,
    "INTJ": 6,
    "NOUN": 7,
    "NUM": 8,
    "PART": 9,
    "PRON": 10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM": 14,
    "VERB": 15,
    "X": 16,
}


def data_reader(data_path):
    """Some useful info
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
                Y[-1].append(token["upos"])
    return X, Y


# Create Dataset Classes
class AtisDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.X, self.Yraw = data_reader(data_path)
        self.Y = []
        self.num_labels = 17
        for ex in self.Yraw:
            self.Y.append([])
            for elem in ex:
                self.Y[-1].append(ATIS_POS_MAPPING[elem])

        assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        X = self.X[ind]
        Y = self.Y[ind]
        Y = torch.as_tensor(Y, dtype=torch.long, device=device)
        return X, Y


def load_atis():
    atis_train = AtisDataset("AtisDataset/en_atis-ud-train.conllu")
    atis_val = AtisDataset("AtisDataset/en_atis-ud-dev.conllu")
    atis_test = AtisDataset("AtisDataset/en_atis-ud-test.conllu")
    return atis_train, atis_val, atis_test
