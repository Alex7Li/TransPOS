"""
Let's try to make a model that generates pseudo labels.

1) Train a model teacher_model on labelled dataset A.

2) Apply teacher_model to unlabelled dataset B and save all predictions
with > 90% confidence as psuedo labels to generate labeled dataset C

4) Train a model student_model on combined dataset A + C

Interesting references
https://openreview.net/forum?id=-ODN6SbiUU

"""
import training
import os
import numpy as np
import torch
from dataloading_utils import get_num_examples
from transformers import get_scheduler
from tqdm import tqdm
import torch.utils.data
from torch.optim import AdamW

class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, pseudolabel_path): 
        loaded = np.load(pseudolabel_path, allow_pickle=True)
        self.inputs = loaded['inputs']
        self.labels = loaded['labels']
          
        assert(len(self.inputs) == len(self.labels))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, ind):
        X = self.inputs[ind]
        Y = self.labels[ind]
        Y = torch.as_tensor(Y, dtype=torch.long, device=self.device) 
        return X, Y

class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2): 
      self.dataset1 = dataset1
      self.dataset2 = dataset2
      self.L1 = len(self.dataset1)
      self.L2 = len(self.dataset2)

    def __len__(self):
        return self.L1 + self.L2

    def __getitem__(self, ind):
      if ind < self.L1:
        return self.dataset1[ind]
      else:
        return self.dataset2[ind - self.L1]


def train_teacher(model_name, dataset_name):
  hparams = {
      'n_epochs': 4,
      'batch_size': 32,
      'dataset': dataset_name,
      'model_name': model_name,
      'save_path': os.path.join('models', 'teacher_' + model_name.split('/')[-1] + "_" + dataset_name),
  }
  training.training_loop(hparams)
  return hparams['save_path']

def get_labels_and_confidence(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    preds = []
    confidences = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        del batch['labels']
        with torch.no_grad():
            outputs = model(**batch)
    
        predictions = torch.argmax(outputs.logits, dim=-1)
        probabilities = torch.exp(outputs.logits)
        confidence = torch.zeros_like(predictions, dtype=torch.float)
        for element_ind in range(predictions.shape[0]):
          for seq_ind in range(predictions.shape[1]):
            confidence[element_ind][seq_ind] = probabilities[element_ind][seq_ind][predictions[element_ind][seq_ind]]
        preds.append(predictions)
        confidences.append(confidence)
    # Contemplate if we can do something about words that are labelled -100
    # (We can't look at the label)
    return preds, confidences

def generate_pseudo_labels(teacher_path, teacher_model_name, teacher_n_labels, dataset_name):
  batch_size = 32
  teacher = training.load_model(teacher_model_name, teacher_n_labels)
  teacher.load_state_dict(torch.load(teacher_path))
  unsupervised_dataset = training.get_dataset(dataset_name, 'train')
  unsupervised_dataloader = training.get_dataloader(teacher_model_name, unsupervised_dataset, shuffle=False, batch_size=batch_size)
  predictions, confidences = get_labels_and_confidence(teacher, unsupervised_dataloader)

  inputs = []
  psuedo_labels = []
  offset = 0
  for (pred_label, confidence, batch) in zip(predictions, confidences, unsupervised_dataloader):
    pred_label = pred_label.detach().cpu().numpy()
    confidence = confidence.detach().cpu().numpy()
    for i in range(len(batch)):
      seq_psuedo_labels = []
      for seq_ind in range(pred_label[i].shape[0]):
        if confidence[i][seq_ind] > .9:
          seq_psuedo_labels.append(pred_label[i][seq_ind])
        else:
          seq_psuedo_labels.append(-100)
      x, _ = unsupervised_dataset[i + offset]
      inputs.append(x)
      psuedo_labels.append(np.stack(seq_psuedo_labels, axis=0))
    offset += len(batch)
  if not os.path.exists('pseudolabels'):
      os.mkdir('pseudolabels')
  save_path = os.path.join('pseudolabels', str(teacher_model_name.split('/')[-1]) + "_" + dataset_name + ".npz")
  np.savez(save_path, inputs=inputs, labels=psuedo_labels)
  return save_path

def train_on_psuedolabels(model_name, pseudolabel_path, base_dataset_name):
  dataset1 = training.get_dataset(base_dataset_name, 'train')
  dataset2 = PseudoDataset(pseudolabel_path)
  dataset = MergedDataset(dataset1, dataset2)
  train_dataloader = training.get_dataloader(model_name, dataset, batch_size=32, shuffle=True)
  val_dataset = training.get_dataset(base_dataset_name, 'val')
  val_dataloader = training.get_dataloader(model_name, val_dataset, batch_size=32, shuffle=False)
  model = training.load_model(model_name, dataset1.num_labels)
  n_epochs = 4
  save_path = os.path.join('models', "student_" + str(model_name.split('/')[-1]) + "_" + base_dataset_name + ".npz")
  training.training_loop(model, train_dataloader, val_dataloader, base_dataset_name, n_epochs, save_path)
  return save_path

def main():
  teacher_model_name = 'bert-large-cased'
  student_model_name = 'bert-large-cased'
  supervised_dataset_name = 'tweebank'
  supervised_dataset_n_labels = 17
  unsupervised_dataset_name = 'TPANN'
  teacher_model_path = train_teacher(teacher_model_name, supervised_dataset_name)
  print(f"Done training. Saved to {teacher_model_path}")
  # teacher_model_path = "models/teacher_bert-large-cased_tweebank"
  pseudolabel_path = generate_pseudo_labels(teacher_model_path, teacher_model_name,\
    supervised_dataset_n_labels, unsupervised_dataset_name)
  print(f"Generated Pseudolabels. Saved to {pseudolabel_path}")
  # pseudolabel_path = "pseudolabels/bert-large-cased_TPANN.npz"
  trained_student_path = train_on_psuedolabels(student_model_name, pseudolabel_path, supervised_dataset_name)
  print(f"Student has been trained, saved to {trained_student_path}")
  # TODO: validate

if __name__ == '__main__':
  main()
  


