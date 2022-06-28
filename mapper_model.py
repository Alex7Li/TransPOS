import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init
from transformers import AutoModel
device = "cuda" if torch.cuda.is_available() else "cpu"

def hardToSoftLabel(hard_label: torch.Tensor, n_classes: int, soft_label_value: torch.Tensor):
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

class Label2LabelDecoder(torch.nn.Module):
    def __init__(self, embedding_dim: int, n_y_labels: int,
                 n_z_labels: int, use_x=True):
        super().__init__()
        self.n_y_labels = n_y_labels
        y_embed_dim = 512
        self.soft_label_value = torch.tensor(4.5).to(device)
        self.ydecoding = nn.Sequential(
            nn.Linear(n_y_labels, y_embed_dim),
            nn.LayerNorm(y_embed_dim)
        )
        rnn_hidden =  512
        n_rnn_layers = 2
        w = torch.empty(2 * n_rnn_layers, rnn_hidden, dtype=torch.float32)
        torch.nn.init.kaiming_normal_(w)
        self.rnn_init = torch.nn.Parameter(w)
        self.register_parameter(name='rnn_init', param=self.rnn_init)
        self.yRNN = torch.nn.GRU(y_embed_dim, rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        xy_hidden = 512
        if use_x:
            self.xydecoding = nn.Sequential(
                nn.Linear(2 * rnn_hidden + embedding_dim, xy_hidden),
                nn.LayerNorm(xy_hidden), nn.GELU(),
                nn.Linear(xy_hidden, n_z_labels),
            )
        else:
            self.xydecoding = nn.Linear(2 * rnn_hidden, n_z_labels)


    def preprocess_label(self, label):
        if len(label.shape) == 2:
            # label is of shape [batch, L]
            label_soft = hardToSoftLabel(label, self.n_y_labels, self.soft_label_value)
        else:
            label_soft = label
        return label_soft.to(device)

    def forward(self, e: torch.Tensor, y: torch.Tensor):
        # e: [batch_size B, sentence_length L, embedding_dim]
        y = self.preprocess_label(y).clone()
        #  y: [B, L, n_labels]
        y = self.ydecoding(y)
        rnn_init = torch.stack([self.rnn_init for _ in range(e.shape[0])], dim=1)
        # rnn_init: [n_layers * 2, B, rnn_hidden]
        assert len(rnn_init.shape) == 3
        y, _ = self.yRNN.forward(y, rnn_init)
        ycat = torch.cat([e, y], dim=2)
        pred_z = self.xydecoding(ycat)
        return pred_z


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
        self.yzdecoding = Label2LabelDecoder(embedding_dim_size, n_y_labels, n_z_labels)
        self.zydecoding = Label2LabelDecoder(embedding_dim_size, n_z_labels, n_y_labels)
        self.ydecoding = nn.Linear(embedding_dim_size, n_y_labels)
        self.zdecoding = nn.Linear(embedding_dim_size, n_z_labels)

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

    def decode_y(self, e: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Combine an embedding with a Y label to predict a Z label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        y: batch of integer label or estimated vector softmax estimate of Y of size n_y_labels.
        """
        return self.yzdecoding(e, y)

    def decode_z(self, e: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Combine an embedding with a Z label to predict a Y label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        z: batch of integer label or estimated vector softmax estimate of Z of size n_z_labels.
        """
        return self.zydecoding(e, z)
