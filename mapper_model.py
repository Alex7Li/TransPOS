import torch
from typing import Union

class MapperModel(torch.nn.Module):
    def __init__(self, base_transformer_name: str, n_y_labels: int, n_z_labels: int):
        self.base_transformer_name = base_transformer_name
        self.n_y_labels = n_y_labels
        self.n_z_labels = n_z_labels

    def encode(self, x: str) -> torch.Tensor:
        """
        Use a transformer to encode x into an embedding dimension E.
        """
        ...

    def decode_y(self, e: torch.Tensor, y: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Combine an embedding with a Y label to predict a Z label.
        e: embedding
        y: integer label or estimated vector softmax estimate of Y of size n_y_labels.
        """
        ...

    def decode_z(self, e: torch.Tensor, z: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Combine an embedding with a Z label to predict a Y label.
        e: embedding
        z: integer label or estimated vector softmax estimate of Z of size n_z_labels.
        """
        ...

