import torch
import training
import torch.nn.functional as F
from typing import Union
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoModel
device = "cuda" if torch.cuda.is_available() else "cpu"

def hardToSoftLabel(hard_label: torch.Tensor, n_classes: int, soft_label_value):
    BS, SEQ_LEN = hard_label.shape
    # Ensure the ignored indicies don't crash the code.
    hard_label = torch.clone(hard_label)
    hard_label[hard_label == -100] += 100
    one_hot = F.one_hot(hard_label, n_classes)
    # Add 5 to the parameter because weight decay will make
    # it want to be zero but really we want it to be positive
    one_hot = one_hot.float().to(device)
    soft_label = one_hot * soft_label_value
    soft_label -= torch.unsqueeze(torch.mean(soft_label, dim=2),2)
    return soft_label

class MapperModel(torch.nn.Module):
    def __init__(self, base_transformer_name: str, n_y_labels: int, n_z_labels: int):
        super().__init__()
        self.base_transformer_name = base_transformer_name
        self.n_y_labels = n_y_labels
        self.n_z_labels = n_z_labels
        self.model = AutoModel.from_pretrained(
           base_transformer_name
        )
        # TODO: How to get this automatically?
        embedding_dim_size = 1024
        if base_transformer_name == 'vinai/bertweet-large':
            embedding_dim_size = 1024
        self.decoderDropout = .5
        decoder_hidden_dim = 512
        decoder_hidden_2_dim = 512
        # Conversion from hard label to soft label
        self.soft_label_value = torch.nn.Parameter(torch.tensor(4.5, dtype=torch.float32))
        self.register_parameter(name='soft_label', param=self.soft_label_value)
        self.yzdecoding_1 = nn.Sequential(
            nn.Linear(n_y_labels, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim), nn.GELU(),
            nn.Dropout(self.decoderDropout),
            )
        self.yzdecoding_2 = nn.Sequential(
            nn.Linear(decoder_hidden_dim + embedding_dim_size, decoder_hidden_2_dim),
            nn.LayerNorm(decoder_hidden_2_dim), nn.GELU(),
            nn.Dropout(self.decoderDropout),
            nn.Linear(decoder_hidden_2_dim, n_z_labels),
        )
        self.zydecoding_1 = nn.Sequential(
            nn.Linear(n_z_labels, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim), nn.GELU(),
            nn.Dropout(self.decoderDropout)
            )
        self.zydecoding_2 = nn.Sequential(
            nn.Linear(decoder_hidden_dim + embedding_dim_size,decoder_hidden_2_dim),
            nn.LayerNorm(decoder_hidden_2_dim), nn.GELU(),
            nn.Dropout(self.decoderDropout),
            nn.Linear(decoder_hidden_2_dim,n_y_labels),
            )
        self.ydecoding = nn.Linear(embedding_dim_size, n_y_labels)
        self.zdecoding = nn.Linear(embedding_dim_size, n_z_labels)

        # Make the soft labels look similar to the hard labels so the model
        # is tricked into thinking they are the same or something
        self.harden_label = True
        self.to(device)


    def encode(self, batch: dict) -> torch.Tensor:
        """
        Use a transformer to encode x into an embedding dimension E.

        output:
        Embeddings of shape
        [batch_size, sentence_length, embedding_dim_size]
        """
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        if 'labels' in batch:
            # Hide the truth!
            labels= batch["labels"]
            del batch["labels"]
            result = self.model(**batch)
            batch["labels"] = labels
        else:
            result = self.model(**batch)
        # Transformer library returns a tuple, in this case
        # it has only 1 element
        return result[0] 

    def preprocess_label(self, label, n_labels):
        if len(label.shape) == 2:
            # label is of shape [batch, L]
            label_soft = hardToSoftLabel(label, n_labels, self.soft_label_value)
        elif len(label.shape) == 3 and self.harden_label:
            # label is of shape [batch, L, n_labels]
            ind = torch.argmax(label, dim=2)
            label_soft = hardToSoftLabel(ind, n_labels, self.soft_label_value)
        else:
            label_soft = label
        return label_soft.to(device)

    def decode_y(self, e: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Combine an embedding with a Y label to predict a Z label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        y: batch of integer label or estimated vector softmax estimate of Y of size n_y_labels.
        """
        y = self.preprocess_label(y, self.n_y_labels).clone()
        y_embed = self.yzdecoding_1(y)
        ycat = torch.cat([e, y_embed], dim=2)
        pred_z = self.yzdecoding_2(ycat)
        return pred_z

    def decode_z(self, e: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Combine an embedding with a Z label to predict a Y label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        z: batch of integer label or estimated vector softmax estimate of Z of size n_z_labels.
        """
        z = self.preprocess_label(z, self.n_z_labels).clone()
        z_embed = self.zydecoding_1(z)
        zcat = torch.cat([e, z_embed], dim=2)
        pred_y = self.zydecoding_2(zcat)
        return pred_y
