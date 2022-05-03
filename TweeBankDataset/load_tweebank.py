import torch
import torch.utils.data
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

twee_train_path = "TweeBankDataset/Tweebank-dev/en-ud-tweet-train.conllu"
twee_dev_path = "TweeBankDataset/Tweebank-dev/en-ud-tweet-dev.conllu"
twee_test_path = "TweeBankDataset/Tweebank-dev/en-ud-tweet-test.conllu"
TWEEBANK_POS_MAPPING = {
    'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5,
    'INTJ': 6, 'NOUN': 7, 'NUM': 8, 'PART': 9, 'PRON': 10, 
    'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'SYM': 14, 'VERB': 15, 'X': 16
}

# Create Dataset Classes
class TweebankTrain(torch.utils.data.Dataset):
    def __init__(self, data_path): 
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