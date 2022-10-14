from loguru import logger

import torch
import torch.nn as nn

import numpy as np

class RandomFeedback(nn.Module):
    r"""Random feedback layer (see paper for further detail)
    """
    def __init__(self, layer_sizes, init="uniform", B_mean_zero=True, Bstd=0.05):
        r"""If normal initialization is chosen, multiple matrices are generated(one per forward matrix).

        Args:
            layer_sizes (list(int)): size of layers of the networt
            init (str, optional): initialization mode (default is 'uniform')
            B_mean_zero (bool, optional): if True, the distribution of the entries is centered around 0 (default is True)
            Bstd (float, optional): standard deviation of the entries of the matrix (default is 0.05)
        """
        super().__init__()

        self.fan_in = layer_sizes[0]
        self.Bstd = Bstd

        self.Bs = []
        if init.lower() == "uniform":
            sd = np.sqrt(6.0 / layer_sizes[-1])
            if B_mean_zero:
                B = (torch.rand(layer_sizes[0], layer_sizes[-1]) * 2 * sd - sd) * Bstd
                self.Bs.append(B)
            else:
                B = (torch.rand(layer_sizes[0], layer_sizes[-1]) * sd) * Bstd
                self.Bs.append(B)
            logger.info(f"Generated feedback matrix with shape {self.Bs[0].shape}")
        elif init.lower() == "normal":
            sd = np.sqrt(2.0 / layer_sizes[0]) * Bstd
            n = len(layer_sizes) - 1
            el = np.prod(layer_sizes[1:-1])
            for i, (size0, size1) in enumerate(zip(layer_sizes, layer_sizes[1:])):
                B = torch.empty(size0, size1).normal_(mean=0, std=(sd / (el ** (1.0 / 2.0))) ** (1.0 / n)
                    )
                self.Bs.append(B)
        else:
            logger.error(f"B initialization '{init.lower()}' is not valid ")
            exit(-1)

    def extra_repr(self):
        return "Bs={}".format(len(self.Bs))

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        r"""Computes projection of input to input space"""
        return input @ self.get_B().T.to(input.device) 

    @torch.no_grad()
    def normalize_B(self):
        r"""Normalizes Bs to keep std constant"""
        std = np.sqrt(2.0 / self.fan_in) * self.Bstd
        for l in range(len(self.Bs)):
            self.Bs[l] *= torch.sqrt(std /  torch.std(self.get_B()))

    @torch.no_grad()
    def get_Bs(self):
        r"""Returns list of Bs"""
        return self.Bs

    @torch.no_grad()
    def get_B(self):
        r"""Returns B (from output to input)"""
        B = self.Bs[0]
        for b in self.Bs[1:]:
            B = B @ b
        return B
