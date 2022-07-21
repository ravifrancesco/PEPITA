from __future__ import barry_as_FLUFL
import enum
from loguru import logger

import torch
import torch.nn as nn

from .FCnet import FCNet

class SkipFCNet(nn.Module):

    # TODO doc
    @torch.no_grad()
    def __init__(self, input_size, output_size, block_sizes, B_mean_zero=True, Bstd=0.05, p=0.1):
        super(SkipFCNet, self).__init__()

        if not block_sizes:
            logger.error('At least one block size is expected')
            exit()

        Bstd = Bstd**(1 / (len(block_sizes) + 1))
        
        self.blocks = [FCNet(
            [input_size] + [block_sizes[0]],
            init='he_normal', B_init='normal',
            B_mean_zero=B_mean_zero,
            Bstd=Bstd,
            p=p,
            final_layer=False)]
        
        self.blocks.extend([FCNet(
            [in_size, out_size], 
            init='he_normal', B_init='normal',
            B_mean_zero=B_mean_zero,
            Bstd=Bstd,
            p=p,
            final_layer=False) for in_size, out_size in zip(block_sizes, block_sizes[1:])])
        
        self.blocks.append(FCNet(
            [block_sizes[-1], output_size], 
            init='he_normal', B_init='normal',
            B_mean_zero=B_mean_zero,
            Bstd= Bstd,
            p=p))

        self.layers_list = []
        
        for block in self.blocks:
            self.layers_list += block.layers
            
        self.layers = nn.Sequential(*self.layers_list)
        
        self.weights = [layer[0].weight for layer in self.layers]
        
    @property
    def activations(self):
        result = []
        list(map(lambda acts: result.extend(acts), self._activations))
        return result
        
    def __repr__(self):
        res = ''
        for b, block in enumerate(self.blocks):
            res += f'Block {b}\n'
            res += block.__repr__() + '\n'
        return res
    
    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)
    
    @torch.no_grad()
    def modulated_forward(self, x, e, batch_size):

        err_l = []
        err = e.clone()
        err_l.insert(0, err.clone())
        for b, block in enumerate(self.blocks[::-1]):
            err = err @ block.B.T
            err_l.insert(0, err.clone())

        l_in = x
        for b, block in enumerate(self.blocks):
            l_in = block.modulated_forward(l_in, err_l[b+1], batch_size)

    @torch.no_grad()
    def get_B(self):
        r"""Returns the total feedback matrix"""
        b = self.blocks[0].get_B()
        for block in self.blocks[1:]:
            b = b @ block.get_B()
        return b

    @torch.no_grad()
    def get_tot_weights(self):
        r"""Returns the total weight matrix (input to output matrix)
        """
        weights = self.weights[0]
        for w in self.weights[1:]:
            weights = w @ weights
        return weights.T

    @torch.no_grad()
    def get_weights_norm(self):
        r"""Returns dict with norms of weight matrixes
        """
        d = {}
        for i, w in enumerate(self.weights):
            d[f'layer{i}'] = torch.linalg.norm(w)
        return d