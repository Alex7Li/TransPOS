import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from dataloading_utils import create_pos_mapping
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def TPANN_data_reader(data_path):
  """  Read txt files"""
  i = 0 
  with open(data_path, "r", encoding="utf-8") as data_file:
    
    X = list()
    Y = list()

    temp_x,temp_y = list(), list()
    for i, line in enumerate(data_file):

      line = line.split()
      if len(line) == 0 and len(temp_x)!= 0:

        X.append(temp_x)
        temp_x = list()
        Y.append(temp_y)
        # update_pos(temp_y)
        # update_pos(temp_y, ignore_punctuation=True)
        temp_y = list()


      elif len(line) == 2:
        temp_x.append(line[0])
        if line[1] == "NONE":
            print(f"{i}: {line}")
        temp_y.append(line[1])

    return X, Y

class TPANNDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        TPANN_POS_TAGS = [':', 'VBZ', 'PRP$', 'WRB', 'MD', 'RB', 'NNS', 'DT', 'UH', 'VBG', ']', 'NN', 'URL', 'VBD', '.', 'VBP', 'POS', 'WP', 'RT', 'VB', 'HT', ')', 'VBN', 'PRP', 'TO', 'NNP', 'JJR', 'USR', 'RP', 'SYM', ',', 'JJ', 'O', 'CC', "''", 'CD', '(', 'PDT', 'IN', '[', 'WDT', 'JJS', 'RBR', 'NNPS', 'LS', 'RBS', 'FW', 'EX']

        self.TPANN_POS_INDEX_MAPPING, self.TPANN_INDEX_POS_MAPPING = create_pos_mapping(TPANN_POS_TAGS)

        self.X, self.Y_pos = TPANN_data_reader(data_path)
        self.Y = []
        self.num_labels = len(TPANN_POS_TAGS)

        for pos_tags in self.Y_pos:
          self.Y.append([])
          for pos_tag in pos_tags:
            self.Y[-1].append(self.TPANN_POS_INDEX_MAPPING[pos_tag])
          
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
        batch_x_pad =pad_sequence(batch_x, batch_first=True, padding_value=0.0)
        
        lengths_y = [y.shape[0] for y in batch_y]
        batch_y_pad = pad_sequence(batch_y, batch_first=True, padding_value=0)
        

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)

def load_tpann():
    train_data = "TPANNDataset/train.txt"
    dev_data = "TPANNDataset/dev.txt"
    test_data = "TPANNDataset/test.txt"
    tpann_train = TPANNDataset(train_data)
    tpann_val = TPANNDataset(dev_data)
    tpann_test = TPANNDataset(test_data)
    return tpann_train, tpann_val, tpann_test
