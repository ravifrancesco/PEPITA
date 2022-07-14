import torch
import torch.nn as nn

class ConsistentDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))

        self.p = p
        self.mask = None

    def extra_repr(self):
        return "p={}".format(self.p)

    def reset_mask(self):
        self.mask = None

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.reset_mask()

    def _get_sample_mask_shape(self, sample_shape):
        return sample_shape

    def _create_mask(self, input):
        mask_shape = input.shape
        mask = torch.empty(mask_shape, dtype=torch.bool).bernoulli_(self.p)
        return mask

    def forward(self, input: torch.Tensor):

        if self.p == 0.0:
            return input
        
        if self.training:
            if self.mask is None:
                self.mask = self._create_mask(input)
            return input*self.mask
        else:
            return (1-self.p)*input
