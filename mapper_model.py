import torch
import training
from typing import Union
from transformers import AutoModelForTokenClassification
device = "cuda" if torch.cuda.is_available() else "cpu"

class MapperModel(torch.nn.Module):
    def __init__(self, base_transformer_name: str, n_y_labels: int, n_z_labels: int,
        embedding_dim_size=128):
        super().__init__()
        self.base_transformer_name = base_transformer_name
        self.n_y_labels = n_y_labels
        self.n_z_labels = n_z_labels
        # TODO: Just use AutoModel rather than
        # AutoModelForTokenClassification, no need to add another layer
        self.model = AutoModelForTokenClassification.from_pretrained(
            base_transformer_name, num_labels=embedding_dim_size
        ).to(device)


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

    def decode_y(self, e: torch.Tensor, y: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Combine an embedding with a Y label to predict a Z label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        y: integer label or estimated vector softmax estimate of Y of size n_y_labels.
        """
        ...

    def decode_z(self, e: torch.Tensor, z: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Combine an embedding with a Z label to predict a Y label.
        e: embedding of shape [batch_size, sentence_length, embedding_dim_size]
        z: integer label or estimated vector softmax estimate of Z of size n_z_labels.
        """
        ...
