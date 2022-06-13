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
    one_hot = one_hot.float().to(device) * (soft_label_value + 5.0)
    # Add some random noise. Maybe it's helpful, maybe not?
    soft_label = torch.normal(
        mean=one_hot, 
        std=.1 * torch.ones(one_hot.shape,
        device=one_hot.device))
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
        self.model.to(device)
        self.decoderDropout = .05
        decoder_hidden_dim = 256
        decoder_hidden_2_dim = 256
        # Conversion from hard label to soft label
        self.soft_label_value = torch.nn.Parameter(torch.tensor(0.0), True)
        self.register_parameter(name='soft_label', param=self.soft_label_value)
        self.yzdecoding = nn.Sequential(
            nn.Linear(embedding_dim_size + n_y_labels, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim), nn.GELU(),
            nn.Dropout(self.decoderDropout),
            nn.Linear(decoder_hidden_dim,decoder_hidden_2_dim),
            nn.LayerNorm(decoder_hidden_2_dim), nn.GELU(),
            nn.Dropout(self.decoderDropout),
            nn.Linear(decoder_hidden_2_dim,n_z_labels),
            )
        self.zydecoding = nn.Sequential(
            nn.Linear(embedding_dim_size + n_z_labels, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim), nn.GELU(),
            nn.Dropout(self.decoderDropout),
            nn.Linear(decoder_hidden_dim,decoder_hidden_2_dim),
            nn.LayerNorm(decoder_hidden_2_dim), nn.GELU(),
            nn.Dropout(self.decoderDropout),
            nn.Linear(decoder_hidden_2_dim,n_y_labels),
            )


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

    def decode_y(self, e: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Combine an embedding with a Y label to predict a Z label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        y: batch of integer label or estimated vector softmax estimate of Y of size n_y_labels.
        """
        if len(y.shape) == 2:
            y = hardToSoftLabel(y, self.n_y_labels, self.soft_label_value)
        y -= torch.unsqueeze(torch.mean(y, dim=2),2)
        y = y.to(device)
        ycat =torch.cat([e, y], dim=2)
        pred_z = self.yzdecoding(ycat)
        return pred_z

    def decode_z(self, e: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Combine an embedding with a Z label to predict a Y label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        z: batch of integer label or estimated vector softmax estimate of Z of size n_z_labels.
        """
        if len(z.shape) == 2:
            z = hardToSoftLabel(z, self.n_z_labels, self.soft_label_value)
        z -= torch.unsqueeze(torch.mean(z, dim=2),2)
        z = z.to(device)
        zcat = torch.cat([e, z], dim=2)
        pred_y = self.zydecoding(zcat)
        return pred_y
