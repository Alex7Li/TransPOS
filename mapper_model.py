from dataclasses import replace
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init
import itertools
from torch.distributions import Categorical
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def hardToSoftLabel(
    hard_label: torch.Tensor, n_classes: int
):
    BS, SEQ_LEN = hard_label.shape
    # Ensure the ignored indicies don't crash the code.
    hard_label = torch.clone(hard_label)
    hard_label[hard_label == -100] += 100
    one_hot = F.one_hot(hard_label, n_classes)
    # Scale one_hot up because weight initalization usually assumes random N(0, 1) distribution for the previous layer
    one_hot = one_hot.float().to(device) * torch.sqrt(torch.tensor(n_classes, dtype=torch.float32, device=hard_label.device))
    one_hot -= torch.unsqueeze(torch.mean(one_hot, dim=2), 2)
    return one_hot


class Label2LabelDecoder(torch.nn.Module):
    def __init__(
        self, embedding_dim: int, n_y_labels: int, n_z_labels: int, use_x=True
    ):
        super().__init__()
        self.n_y_labels = n_y_labels
        y_embed_dim = 512
        self.ydecoding = nn.Sequential(
            nn.Linear(n_y_labels, y_embed_dim), nn.LayerNorm(y_embed_dim)
        )
        rnn_hidden = 512
        n_rnn_layers = 2
        w = torch.empty(2 * n_rnn_layers, rnn_hidden, dtype=torch.float32)
        torch.nn.init.kaiming_normal_(w)
        self.rnn_init = torch.nn.Parameter(w)
        self.register_parameter(name="rnn_init", param=self.rnn_init)
        self.yRNN = torch.nn.GRU(
            y_embed_dim, rnn_hidden, num_layers=2, batch_first=True, bidirectional=True
        )
        xy_hidden = 512
        self.use_x = use_x
        if use_x:
            # Huge so that the second layer can't just rely on x!
            self.x_dropout = nn.Dropout(p=.88, inplace=True)
            self.xydecoding = nn.Sequential(
                nn.Linear(2 * rnn_hidden + embedding_dim, xy_hidden),
                nn.LayerNorm(xy_hidden),
                nn.GELU(),
                nn.Linear(xy_hidden, n_z_labels),
            )
        else:
            self.xydecoding = nn.Linear(2 * rnn_hidden, n_z_labels)

    def preprocess_label(self, label):
        if len(label.shape) == 3:
            # label is of shape [batch, L, self.n_y_labels]
            label = Categorical(logits=label).sample()
        # label is of shape [batch, L]
        label_soft = hardToSoftLabel(label, self.n_y_labels)
        # label_soft shape is [batch, L, self.n_y_labels]
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
        if self.use_x:
            e = self.x_dropout(e)
            ycat = torch.cat([e, y], dim=2)
        else:
            ycat = y
        pred_z = self.xydecoding(ycat)
        return pred_z


class MapperModel(torch.nn.Module):
    def __init__(self, base_transformer_name: str, n_y_labels: int, n_z_labels: int, parameters):
        super().__init__()
        self.base_transformer_name = base_transformer_name
        self.n_y_labels = n_y_labels
        self.n_z_labels = n_z_labels
        self.model_y = AutoModel.from_pretrained(base_transformer_name)
        if parameters.use_shared_encoder:
            self.model_z = self.model_y
        else:
            self.model_z = AutoModel.from_pretrained(base_transformer_name)
        embedding_dim_size = 1024
        if base_transformer_name == "vinai/bertweet-large":
            embedding_dim_size = 1024
        elif base_transformer_name == "gpt2":
            embedding_dim_size = 768
        self.yzdecoding = Label2LabelDecoder(
            embedding_dim_size, n_y_labels, n_z_labels, parameters.decoder_use_x)
        self.zydecoding = Label2LabelDecoder(
            embedding_dim_size, n_z_labels, n_y_labels, parameters.decoder_use_x)
        self.ydecoding = nn.Sequential(
            nn.Linear(embedding_dim_size, n_y_labels),
        )
        self.zdecoding = nn.Sequential(
            nn.Linear(embedding_dim_size, n_z_labels),
        )
        if parameters.use_shared_encoder:
            self.pretrained_params = itertools.chain(
                self.model_y.parameters(),
            )
        else:
            self.pretrained_params = itertools.chain(
                self.model_y.parameters(),
                self.model_z.parameters(),
            )
        self.auxilary_params = itertools.chain(
            self.yzdecoding.parameters(),
            self.zydecoding.parameters(),
            self.ydecoding.parameters(),
            self.zdecoding.parameters(),
        )
        self.to(device)

    def encode(self, batch: dict, output_type='y') -> torch.Tensor:
        """
        Use a transformer to encode x into an embedding dimension E.

        output:
        Embeddings of shape
        [batch_size, sentence_length, embedding_dim_size]
        """
        model = self.model_y if output_type == 'y' else self.model_z
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        if "labels" in batch:
            # Hide the truth!
            labels = batch["labels"]
            del batch["labels"]
            result = model(**batch)
            batch["labels"] = labels
        else:
            result = model(**batch)
        # Transformer library returns a tuple, in this case
        # it has only 1 element
        return result[0]

    def encode_y(self, batch) -> torch.Tensor:
        return self.encode(batch, 'y')

    def encode_z(self, batch) -> torch.Tensor:
        return self.encode(batch, 'z')

    def decode_z(self, e_z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Combine an embedding with a Y label to predict a Z label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        y: batch of integer label or estimated vector softmax estimate of Y of size n_y_labels.
        """
        return self.yzdecoding(e_z, y)

    def decode_y(self, e_y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Combine an embedding with a Z label to predict a Y label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        z: batch of integer label or estimated vector softmax estimate of Z of size n_z_labels.
        """
        return self.zydecoding(e_y, z)
