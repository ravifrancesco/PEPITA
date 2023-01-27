import torch
import torch.nn as nn

class Normalization(nn.Module):
    r"""Normalization layer: output is the normalized input
    """
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X / torch.linalg.norm(X)