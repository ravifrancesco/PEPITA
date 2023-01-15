import torch
import torch.nn as nn

class Normalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X / torch.linalg.norm(X)