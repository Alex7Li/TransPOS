from typing import *
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from dataloading_utils import create_pos_mapping
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Ark twitter annotation guidelines --> https://github.com/brendano/ark-tweet-nlp/blob/master/docs/annot_guidelines.pdf (25 tags)
# ARK Twitter dataset

def read_ark_file(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
  """
  Process a data file from ARK twitter into words and their POS tags
  The word at the i-th index corresponds to the i-th POS
  --------------------------------------------------------------------
  Args
  -----
  file_path: str
    The full path to the data file

  Returns
  --------
  X: The words in the data file

  Y: The POS tags in the data file
  """
  with open(file_path, "r") as f:
    lines = f.read().splitlines()
    lines = [line.split("\t") for line in lines]
    X = [[]]
    Y = [[]]
    for line in lines:
      if len(line) != 2:
        X.append([])
        Y.append([])
        continue

      X[-1].append(line[0])
      Y[-1].append(line[1])

  return X, Y
def read_ark_file_test(ark_train_path):
  train_X, train_Y = read_ark_file(ark_train_path)
  assert train_X[1][1] == '@TheBlissfulChef'
  assert train_Y[2][0] == '!'

def ark_pos_mapping_test():
  ark_pos_tags = ["N", "O", "^", "S", "Z", "V", "A", "R",
                "!", "D", "P", "&", "T", "X", "#", "@",
                "~", "U", "E", "$", ",", "G", "L", "M", "Y"
                ]
  ARK_POS_INDEX_MAPPING, ARK_INDEX_POS_MAPPING = create_pos_mapping(ark_pos_tags)
  assert len(ARK_POS_INDEX_MAPPING) == 25
  assert len(ARK_INDEX_POS_MAPPING) == 25
  assert ARK_POS_INDEX_MAPPING["L"] == 22
  assert ARK_INDEX_POS_MAPPING[7] == "R"


class ArkDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        ARK_POS_TAGS = ["N", "O", "^", "S", "Z", "V", "A", "R",
                "!", "D", "P", "&", "T", "X", "#", "@",
                "~", "U", "E", "$", ",", "G", "L", "M", "Y"
                ]

        ARK_POS_INDEX_MAPPING, ARK_INDEX_POS_MAPPING = create_pos_mapping(ARK_POS_TAGS)
        self.ARK_POS_INDEX_MAPPING = ARK_POS_INDEX_MAPPING
        self.ARK_INDEX_POS_MAPPING = ARK_INDEX_POS_MAPPING

        self.X, self.Y_pos = read_ark_file(data_path)
        self.Y = []
        self.num_labels = len(ARK_POS_TAGS)

        for pos_tags in self.Y_pos:
          self.Y.append([])
          for pos_tag in pos_tags:
            self.Y[-1].append(self.ARK_POS_INDEX_MAPPING[pos_tag])
          
        assert(len(self.X) == len(self.Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        x = self.X[ind]
        y = self.Y[ind]
        y = torch.as_tensor(y, dtype=torch.long, device=device) 
        return x, y

    def collate_fn(batch):
        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        # convert all to tensors
        batch_x = [torch.as_tensor(x,dtype=torch.float64,device=device) for x,y in batch]
        batch_Y = [torch.as_tensor(y,dtype=torch.long,device=device) for x,y in batch]
        

        lengths_x = [x.shape[0] for x in batch_x]
        batch_x_pad = pad_sequence(batch_x, batch_first=True, padding_value=0.0)
        
        lengths_y = [y.shape[0] for y in batch_y]
        batch_y_pad = pad_sequence(batch_y, batch_first=True, padding_value=0)
        

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)

def load_ark():
    ark_train_path = "ArkDataset/daily547.conll"
    ark_val_path = "ArkDataset/oct27.traindev"
    ark_test_path = "ArkDataset/oct27.test"
    read_ark_file_test(ark_train_path)
    ark_pos_mapping_test()
    ark_train = ArkDataset(ark_train_path)
    ark_val = ArkDataset(ark_val_path)
    ark_test = ArkDataset(ark_test_path)
    return ark_train, ark_val, ark_test
