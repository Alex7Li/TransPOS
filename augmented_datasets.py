

import torch
import numpy as np
import itertools
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from dataloading_utils import create_pos_mapping
from conllu import parse_incr
from ArkDataset.load_ark import load_ark
from TPANNDataset.load_tpann import load_tpann
from TweeBankDataset.load_tweebank import load_tweebank
from AtisDataset.load_atis import load_atis
from GUMDataset.load_GUM import load_gum
from nltk.corpus import wordnet as wn
from dataloading_utils import filter_negative_hundred, TransformerCompatDataset, get_num_examples, get_validation_acc, get_dataset_mapping
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, BertForTokenClassification, AutoTokenizer, get_scheduler
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TWEEBANK_POS_MAPPING = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5, 'INTJ': 6, 'NOUN': 7, 'NUM': 8, 'PART': 9, 'PRON': 10, 'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'SYM': 14, 'VERB': 15, 'X': 16}

ATIS_POS_MAPPING = {
  'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5,
  'INTJ': 6, 'NOUN': 7, 'NUM': 8, 'PART': 9, 'PRON': 10,
  'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'SYM': 14, 'VERB': 15, 'X': 16
}
GUM_POS_MAPPING = {
  'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5,
  'INTJ': 6, 'NOUN': 7, 'NUM': 8, 'PART': 9, 'PRON': 10,
  'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'SYM': 14, 'VERB': 15, 'X': 16
}

class ArkAugDataset(torch.utils.data.Dataset):
    def __init__(self, X,Y):

        ARK_POS_TAGS = ["N", "O", "^", "S", "Z", "V", "A", "R",
                "!", "D", "P", "&", "T", "X", "#", "@",
                "~", "U", "E", "$", ",", "G", "L", "M", "Y"
                ]

        ARK_POS_INDEX_MAPPING, ARK_INDEX_POS_MAPPING = create_pos_mapping(ARK_POS_TAGS)
        self.ARK_POS_INDEX_MAPPING = ARK_POS_INDEX_MAPPING
        self.ARK_INDEX_POS_MAPPING = ARK_INDEX_POS_MAPPING

        self.X, self.Y_pos = X,Y
        self.num_labels = len(ARK_POS_TAGS)
        self.Y = []
        for pos_tags in self.Y_pos:
          self.Y.append([])
          for pos_tag in pos_tags:
            self.Y[-1].append(self.ARK_POS_INDEX_MAPPING[pos_tag])
        assert(len(self.X) == len(self.Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        # x = self.X[ind][0]
        # xhat = self.X[ind][1]
        y = self.Y[ind]
        y = torch.as_tensor(y, dtype=torch.long, device=device) 
        return  list(itertools.chain(*self.X[ind])), y.repeat(2)

    def collate_fn(batch):
        batch_x = [x for x,y in batch]
        batch_y = [y for x,xhat,y in batch]

        # convert all to tensors
        batch_x = [torch.as_tensor(x,dtype=torch.float64,device=device) for x,y in batch]
        # batch_xhat = [torch.as_tensor(xhat,dtype=torch.float64,device=device) for x,y in batch]
        batch_Y = [torch.as_tensor(y,dtype=torch.long,device=device) for x,y in batch]
        

        lengths_x = [x.shape[0] for x in batch_x]
        batch_x_pad = pad_sequence(batch_x, batch_first=True, padding_value=0.0)
        # batch_xhat_pad = pad_sequence(batch_xhat, batch_first=True, padding_value=0.0)
        
        lengths_y = [y.shape[0] for y in batch_y]
        batch_y_pad = pad_sequence(batch_y, batch_first=True, padding_value=0)
        

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)



class TPANNAugDataset(torch.utils.data.Dataset):
    def __init__(self, X,Y):
        self.X, self.Y_pos = X,Y
        TPANN_POS_TAGS = [':', 'VBZ', 'PRP$', 'WRB', 'MD', 'RB', 'NNS', 'DT', 'UH', 'VBG', ']', 'NN', 'URL', 'VBD', '.', 'VBP', 'POS', 'WP', 'RT', 'VB', 'HT', ')', 'VBN', 'PRP', 'TO', 'NNP', 'JJR', 'USR', 'RP', 'SYM', ',', 'JJ', 'O', 'CC', "''", 'CD', '(', 'PDT', 'IN', '[', 'WDT', 'JJS', 'RBR', 'NNPS', 'LS', 'RBS', 'FW', 'EX']
        self.TPANN_POS_INDEX_MAPPING, self.TPANN_INDEX_POS_MAPPING = create_pos_mapping(TPANN_POS_TAGS)
        self.Y = []
        for sentence in self.Y_pos:
            label = []
            for word in sentence:
                label.append(int(self.TPANN_POS_INDEX_MAPPING[word]))
            self.Y.append(label)

        assert(len(self.X) == len(self.Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        y = self.Y[ind]
        y = torch.as_tensor(y, dtype=torch.long, device=device) 
        
        return list(itertools.chain(*self.X[ind])), y.repeat(2)

    def collate_fn(batch):
        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        # convert all to tensors
        batch_x = [torch.as_tensor(x,dtype=torch.float64,device=device) for x,y in batch]
        batch_Y = [torch.as_tensor(y,dtype=torch.long,device=device) for x,y in batch]
        

        lengths_x = [x.shape[0] for x in batch_x]
        batch_x_pad =pad_sequence(batch_x, batch_first=True, padding_value=0.0)
        
        lengths_y = [y.shape[0] for y in batch_y]
        batch_y_pad = pad_sequence(batch_y, batch_first=True, padding_value=0)
        

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)
# Create Dataset Classes
class AtisAugDataset(torch.utils.data.Dataset):
    def __init__(self, X,Y): 
        self.X, self.Yraw = X,Y
        self.Y = []
        for ex in self.Yraw:
          self.Y.append([])
          for elem in ex:
            self.Y[-1].append(ATIS_POS_MAPPING[elem])
          
        assert(len(self.X) == len(self.Yraw))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        Y = self.Y[ind]
        Y = torch.as_tensor(Y, dtype=torch.long, device=device) 
        return list(itertools.chain(*self.X[ind])), Y.repeat(2)

class GUMAugDataset(torch.utils.data.Dataset):
    def __init__(self, X,Y): 
        self.X, self.Yraw = X,Y
        self.Y = []
        for ex in self.Yraw:
          self.Y.append([])
          for elem in ex:
            if elem == '_':
                elem = 'X'
            self.Y[-1].append(GUM_POS_MAPPING[elem])
          
        assert(len(self.X) == len(self.Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        Y = self.Y[ind]
        Y = torch.as_tensor(Y, dtype=torch.long, device=device) 
        return list(itertools.chain(*self.X[ind])), Y.repeat(2)
def create_pos_mapping(ark_pos_tags):
  """
  Creates two dictionaries
  - The pos tags are mapped to their indices in the list and vice versa
  ----------------------------------------------------------------------
  Args
  -----
  ark_pos_tags: List[str]
    The list of pos tags used in the ark dataset

  Returns
  --------
  ark_pos_index_mapping: Dict[str, int]
    The mapping of pos to their indices

  ark_index_pos_mapping: Dict[int, str]
    The mapping of the indices to their corresponding pos
  """
  ark_pos_index_mapping = {tag:i for i, tag in enumerate(ark_pos_tags)}
  ark_index_pos_mapping = {v:k for k, v in ark_pos_index_mapping.items()}
  return ark_pos_index_mapping, ark_index_pos_mapping

class TweebankAugTrain(torch.utils.data.Dataset):
    def __init__(self, X,Y): 
        self.X, self.Yraw = X,Y
        self.Y = []
        for ex in self.Yraw:
          self.Y.append([])
          for elem in ex:
            self.Y[-1].append(TWEEBANK_POS_MAPPING[elem])
          
        assert(len(self.X) == len(self.Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        Y = self.Y[ind]
        Y = torch.as_tensor(Y, dtype=torch.long, device=device) 
        return list(itertools.chain(*self.X[ind])), Y.repeat(2)
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

def generate_mask_and_data(x_input,input_type,mid=None):
    batch_x_n = []
    batch_xhat = []
    for i,x in enumerate(x_input):
        data_len = torch.numel(x)
        xhat = x.clone()
        if mid is None:
            if torch.numel(((x.flatten() == 50256).nonzero(as_tuple=True))[0]) == 0:
                idx = len(x.flatten())
            else:
                idx  = ((x.flatten() == 50256).nonzero(as_tuple=True)[0])[0].item()
            mid = idx//2 
        if input_type == "attention":
            val = 0 
        if input_type =="input":
            val = 50256
        if input_type=="labels":
            val = -100
        x_n = x[:mid+2].detach().cpu().numpy()
        xhat = x[mid:].detach().cpu().numpy()
        x_n = np.pad(x_n,(0,data_len - len(x_n)),constant_values=(val))
        xhat = np.pad(xhat,(0,data_len - len(xhat)),constant_values=(val))
        x_n = torch.from_numpy(x_n)
        xhat = torch.from_numpy(xhat)
        batch_x_n.append(x_n.tolist())
        batch_xhat.append(xhat.tolist())
    batch_x_n = torch.tensor(batch_x_n).cuda() if torch.cuda.is_available()  else torch.tensor(batch_x_n)
    batch_xhat = torch.tensor(batch_xhat).cuda() if torch.cuda.is_available()  else torch.tensor(batch_xhat)
    return batch_x_n, batch_xhat,mid
 

# generate_mask(x,"attention")
def get_dataloader(model_name, dataset, batch_size, shuffle=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, use_fast=True)
    if model_name == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compat_dataset = TransformerCompatDataset(dataset, tokenizer)
    dataloader = DataLoader(compat_dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=data_collator)
    return dataloader

def get_augmented_dataset(train_X,train_Y):
  augmented_examples = []
  augmented_labels = []
  augment_percent = .1
  for i,sentence in enumerate(train_X):
    break1 = False
    breaktonextword = False
    break2 = False
    break3 = False
    augmented = False
    change_made = False
    ex = sentence.copy()
    num_words_to_augment = max(1,int(augment_percent*(len(sentence))))
    for j,word in enumerate(sentence):
        temp_pos = ""
        if train_Y[i][j] in ["V","N","A","VB","VBD","VBG","VBN","VBP",
                            "VBZ","JJ","JJS","JJR","NN","NNS","ADJ","NOUN","VERB"]:
            if train_Y[i][j] in ["VB","VBD","VBG","VBN","VBP","VBZ","VERB"]:
                temp_pos = "V"

            if train_Y[i][j] in ["JJ","JJS","JJR","ADJ"]:
                temp_pos = "A"
            if train_Y[i][j] in ["NN","NNS","NOUN"]:
                temp_pos = "N"
            for s,synset in enumerate(wn.synsets(train_X[i][j])):
                if s==0:
                    continue
                if synset.pos() == train_Y[i][j].lower() or synset.pos() ==temp_pos.lower():
                    for lemma in synset.lemmas():
                        ex[j] = lemma.name()
                        change_made=True
                        num_words_to_augment-=1
                        if num_words_to_augment ==0:
                            augmented_examples.append([ex,train_X[i]])
                            augmented_labels.append(train_Y[i])
                            break1 = True
                            augmented = True
                            break
                        elif j==(len(sentence)-1) and change_made:
                            augmented_examples.append([ex,train_X[i]])
                            augmented_labels.append(train_Y[i])
                            augmented = True
                            break1 = True
                            break
                        elif num_words_to_augment>0:
                            breaktonextword = True
                            break
                if breaktonextword:
                    breaktonextword = False
                    break
                if break1:
                    break1 = False
                    break2 = True
                    break
            if break2:
                break2 = False
                break
  return augmented_examples, augmented_labels

def get_augmented_dataloader(dataset="ark",partition="train",model="gpt2"):
    """
    Dataset can be [ark, tpann,twee, atis,gum]
    partition can be [train,dev,test]
    """
    TPANN_POS_TAGS = [':', 'VBZ', 'PRP$', 'WRB', 'MD', 'RB', 'NNS', 'DT', 'UH', 'VBG', ']', 'NN', 'URL', 'VBD', '.', 'VBP', 'POS', 'WP', 'RT', 'VB', 'HT', ')', 'VBN', 'PRP', 'TO', 'NNP', 'JJR', 'USR', 'RP', 'SYM', ',', 'JJ', 'O', 'CC', "''", 'CD', '(', 'PDT', 'IN', '[', 'WDT', 'JJS', 'RBR', 'NNPS', 'LS', 'RBS', 'FW', 'EX']
    TPANN_POS_INDEX_MAPPING, TPANN_INDEX_POS_MAPPING = create_pos_mapping(TPANN_POS_TAGS)
    if dataset.lower() == "ark":
        ark_train_x, ark_train_y = load_ark(return_lists=True,partition=partition)
        ark_aug_train_x, ark_aug_train_y = get_augmented_dataset(ark_train_x,ark_train_y)
        ark_aug_train = ArkAugDataset(ark_aug_train_x, ark_aug_train_y)
        ark_aug_dataloader = get_dataloader(model,ark_aug_train,20,shuffle=False )
        return ark_aug_dataloader
    if dataset.lower() =="tpann":
        tpann_train, tpann_val, tpann_test = load_tpann()
        tpann_train_x = []
        tpann_train_raw_y =[]
        for elem in tpann_train:
            tpann_train_x.append(elem[0])
            tpann_train_raw_y.append(elem[1].tolist())
        
        tpann_train_y = []
        for elem in tpann_train_raw_y:
            ex = []
            for index_ in elem:
                candidate_mapping = TPANN_INDEX_POS_MAPPING[index_]
                ex.append(candidate_mapping)
            tpann_train_y.append(ex)
        tpann_aug_train_x, tpann_aug_train_y = get_augmented_dataset(tpann_train_x,tpann_train_y)
        assert len(tpann_aug_train_x) == len(tpann_aug_train_y)
        tpann_aug_train = TPANNAugDataset(tpann_aug_train_x, tpann_aug_train_y)
        tpann_aug_dataloader = get_dataloader(model,tpann_aug_train,20,shuffle=False)
        return tpann_aug_dataloader
    if dataset.lower()=="atis":
        atis_train_x, atis_train_y = data_reader(f"AtisDataset/en_atis-ud-{partition}.conllu")
        atis_aug_train_x,atis_aug_train_y = get_augmented_dataset(atis_train_x,atis_train_y)
        atis_aug_train = AtisAugDataset(atis_aug_train_x,atis_aug_train_y)
        atis_aug_dataloader = get_dataloader(model,atis_aug_train,20,shuffle=False)
        return atis_aug_dataloader
    if dataset.lower()=="gum":
        gum_train_x, gum_train_y = data_reader(f"GUMDataset/en_gum-ud-{partition}.conllu")
        gum_aug_train_x,gum_aug_train_y = get_augmented_dataset(gum_train_x,gum_train_y)
        gum_aug_train = GUMAugDataset(gum_aug_train_x,gum_aug_train_y)
        gum_aug_dataloader = get_dataloader(model,gum_aug_train,20,shuffle=False)
        return gum_aug_dataloader

    if dataset.lower() =="tweebank":
        twee_train_x, twee_train_y = data_reader(f"TweeBankDataset/Tweebank-dev/en-ud-tweet-{partition}.conllu")
        twee_aug_train_x,twee_aug_train_y = get_augmented_dataset(twee_train_x,twee_train_y)
        twee_aug_train = TweebankAugTrain(twee_aug_train_x,twee_aug_train_y)
        twee_aug_dataloader = get_dataloader(model,twee_aug_train,20,shuffle=False)
        return twee_aug_dataloader
### Test operation
test_ark_aug = False
if test_ark_aug:
    ark_train_x, ark_train_y = load_ark(return_lists=True,partition="train")
    ark_aug_train_x, ark_aug_train_y = get_augmented_dataset(ark_train_x,ark_train_y)
    ark_aug_train = ArkAugDataset(ark_aug_train_x, ark_aug_train_y)
    ark_aug_dataloader = get_dataloader("gpt2",ark_aug_train,20,shuffle=False )
    for batch in ark_aug_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        print(batch)
        input("")
    print("data loader works!")
    # input("")
if __name__ == "__main__":
    test_tpann_aug = False
    if test_tpann_aug:
        tpann_train, tpann_val, tpann_test = load_tpann()
        tpann_train_x = []
        tpann_train_raw_y =[]
        for elem in tpann_train:
            tpann_train_x.append(elem[0])
            tpann_train_raw_y.append(elem[1].tolist())
        TPANN_POS_TAGS = [':', 'VBZ', 'PRP$', 'WRB', 'MD', 'RB', 'NNS', 'DT', 'UH', 'VBG', ']', 'NN', 'URL', 'VBD', '.', 'VBP', 'POS', 'WP', 'RT', 'VB', 'HT', ')', 'VBN', 'PRP', 'TO', 'NNP', 'JJR', 'USR', 'RP', 'SYM', ',', 'JJ', 'O', 'CC', "''", 'CD', '(', 'PDT', 'IN', '[', 'WDT', 'JJS', 'RBR', 'NNPS', 'LS', 'RBS', 'FW', 'EX']
        TPANN_POS_INDEX_MAPPING, TPANN_INDEX_POS_MAPPING = create_pos_mapping(TPANN_POS_TAGS)
        tpann_train_y = []
        for elem in tpann_train_raw_y:
            ex = []
            for index_ in elem:
                candidate_mapping = TPANN_INDEX_POS_MAPPING[index_]
                ex.append(candidate_mapping)
            tpann_train_y.append(ex)
        
        tpann_aug_train_x, tpann_aug_train_y = get_augmented_dataset(tpann_train_x,tpann_train_y)
        assert len(tpann_aug_train_x) == len(tpann_aug_train_y)
        
        tpann_aug_train = TPANNAugDataset(tpann_aug_train_x, tpann_aug_train_y)
        tpann_aug_dataloader = get_dataloader("gpt2",tpann_aug_train,20,shuffle=False)
        for batch in tpann_aug_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            print(batch)
            break
        print("data loader works!")
        input("")
    test_atis= False
    if test_atis:
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
        atis_train_x, atis_train_y = data_reader("AtisDataset/en_atis-ud-train.conllu")
        atis_aug_train_x,atis_aug_train_y = get_augmented_dataset(atis_train_x,atis_train_y)
        atis_aug_train = AtisAugDataset(atis_aug_train_x,atis_aug_train_y)
        atis_aug_dataloader = get_dataloader("gpt2",atis_aug_train,20,shuffle=False)
        for batch in atis_aug_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            print(batch)
            break
        input("")
    test_gum = False
    if test_gum:
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
        gum_train_x, gum_train_y = data_reader("GUMDataset/en_gum-ud-train.conllu")
        gum_aug_train_x,gum_aug_train_y = get_augmented_dataset(gum_train_x,gum_train_y)
        gum_aug_train = GUMAugDataset(gum_aug_train_x,gum_aug_train_y)
        gum_aug_dataloader = get_dataloader("gpt2",gum_aug_train,20,shuffle=False)
        for batch in gum_aug_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            print(batch)
            break
        input("")
    test_twee = False
    if test_twee:
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
        twee_train_x, twee_train_y = data_reader("TweeBankDataset/Tweebank-dev/en-ud-tweet-train.conllu")
        twee_aug_train_x,twee_aug_train_y = get_augmented_dataset(twee_train_x,twee_train_y)
        twee_aug_train = TweebankAugTrain(twee_aug_train_x,twee_aug_train_y)
        twee_aug_dataloader = get_dataloader("gpt2",twee_aug_train,20,shuffle=False)
        for batch in twee_aug_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            print(batch)
            break
        print("twee dataloader works")
        input("")