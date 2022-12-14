import torch
import torch.nn as nn

class ConsistentDropout(nn.Module):
    r"""Dropout Layer that requires manual reset of the mask
    """
    def __init__(self, p=0.5):
        r"""
        Args:
            p (float, optional): dropout probability (default is 0.5)
        """
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))

        self.p = p
        self.mask = None

    def extra_repr(self):
        return "p={}".format(self.p)

    def reset_mask(self):
        r"""Reset dropout mask
        """
        self.mask = None

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.reset_mask()

    def _get_sample_mask_shape(self, sample_shape):
        r"""Returns mask shape

        Args:
            sample_shape (shape): sample shape

        Returns:
            sample shape (shape)
        """
        return sample_shape

    def _create_mask(self, input):
        r"""Creates the mask by drawing from Bernoulli distribution with probability p

        Args:
            input (torch.Tensor): input
        
        Returns:
            mask (torch.Tensor): dropout mask
        """
        mask_shape = input.shape
        mask = torch.empty(mask_shape, dtype=torch.bool).bernoulli_(1-self.p) / (1-self.p)
        return mask

    def forward(self, input: torch.Tensor):

        if self.p == 0.0:
            return input
        
        if self.training:
            if self.mask is None:
                self.mask = self._create_mask(input)
            self.mask = self.mask.to(input.device)
            return torch.mul(input, self.mask)
        else:
            return input


class ConsistentDropoutNd(ConsistentDropout):
    def _create_mask(self, input):
        r"""Creates the mask by drawing from Bernoulli distribution with probability p

        Args:
            input (torch.Tensor): input

        Returns:
            mask (torch.Tensor): dropout mask
        """
        mask_shape = list(input.shape)
        mask_shape[2:] = [1] * len(mask_shape[2:])
        mask = torch.empty(mask_shape, dtype=torch.bool).bernoulli_(1-self.p) / (1-self.p)
        return mask
