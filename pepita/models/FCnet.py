import numpy as np

from loguru import logger

import torch
import torch.nn as nn

from pepita.models.layers.ConsistentDropout import ConsistentDropout

# code adapted from https://github.com/avicooper1/CPSC490/blob/main/pepita.ipynb

@torch.no_grad()
def generate_layer(in_size, out_size, p=0.1):
    r"""Helper function to generate a Fully Connected block

    Args:
        in_size (int): input size
        out_size (int): output size
        p (float): dropout rate (default is 0.1)
    
    Returns:
        Fully connected block (nn.Sequential): fully connected block of the specified dimensions
    """
    w = nn.Linear(in_size, out_size, bias=False)
    d = ConsistentDropout(p=p)
    a = nn.ReLU()
    
    layer_limit = np.sqrt(6.0 / in_size) # TODO change initialization to include sqrt(6)
    #torch.nn.init.kaiming_uniform_(w.weight, mode='fan_in', nonlinearity='relu')
    torch.nn.init.uniform_(w.weight, a=-layer_limit, b=layer_limit)
    
    return nn.Sequential(w, d, a)

@torch.no_grad()
def collect_activations(model, l):
    r"""Return hook for registering the given layer activations
    
    Args:
        model (torch.nn.Module): model of which activations are collected
        l (int): index of the layer of the model

    Returns:
        hook (callable): hook to collect the activations
    """
    def hook(self, input, output):
        model.activations[l] = output.detach()
    return hook

def generate_B(n_in, n_out, B_mean_zero=True, Bstd=0.05):
    r"""Helper function to generate the feedback matrix

    Args:
        n_in (int): network input size
        n_out (int): network output size
        B_mean_zero (bool, optional): if True, the distribution of the entries is centered around 0 (default is True)
        Bstd (float, optional): standard deviation of the entries of the matrix (default is 0.05)
    
    Returns:
        Tensor (torch.Tensor): the feedback matrix
    """
    sd = np.sqrt(6 / n_in)
    if B_mean_zero:
        B = (torch.rand(n_in, n_out) * 2 * sd - sd) * Bstd  # mean zero
    else:
        B = (torch.rand(n_in, n_out) * sd) * Bstd
        
    return B

class FCNet(nn.Module):
    r"""Fully connected network

    Attributes:
        layers (nn.Module): layers of the network
        B (torch.Tensor): feedback matrix
    """
    @torch.no_grad()
    def __init__(self, layer_sizes, B_mean_zero=True, Bstd=0.05, p=0.1, final_layer=True):
        r"""
        Args:
            layer_sizes (list[int]): sizes of each layers
            B_mean_zero (bool, optional): if True, the distribution of the entries is centered around 0 (default is True)
            Bstd (float, optional): standard deviation of the entries of the matrix (default is 0.05)
            p (float, optional): dropout rate (default is 0.1) 
            final_layer (bool, optional): if True, the last layer is treated as the last layer of the network (default is True)
        """
        super(FCNet, self).__init__()

        self.layers_list = [generate_layer(in_size, out_size, p) for in_size, out_size in zip(layer_sizes, layer_sizes[1:-1])]
        self.layers_list.append(nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=False), nn.Softmax(dim=1) if final_layer else nn.ReLU()))
        
        self.layers = nn.Sequential(*self.layers_list)
        
        self.weights = [layer[0].weight for layer in self.layers]
        
        self.activations = [None] * len(self.layers)
        
        for l, layer in enumerate(self.layers):
            layer.register_forward_hook(collect_activations(self, l))
            
        self.B = generate_B(layer_sizes[0], layer_sizes[-1], B_mean_zero=B_mean_zero, Bstd=Bstd)
        logger.info(f'Generated feedback matrix with shape {self.B.shape}')
 
        
        # if torch.cuda.is_available():
        #     self.to('cuda')
        #     logger.info(f'Feedback matrix moved to cuda')
        
    @torch.no_grad()
    def get_activations(self):
        r"""Returns the activations of the network
        
        Returns:
            activations (list[torch.Tensor]): the network activations
        """
        return [activations.clone() for activations in self.activations]

    def reset_dropout_masks(self):
        # TODO doc
        for module in self.modules():
            if module.__class__ is ConsistentDropout:
                module.reset_mask()
        
    @torch.no_grad()        
    def forward(self, x):
        r"""Computes the forward pass and returns the output
        
        Args:
            x (torch.Tensor): the input

        Returns:
            output (torch.Tensor): the network output
        """
        return self.layers(x)

    @torch.no_grad()
    def update(self, x, e, lr, batch_size, first_block=True, last_block=True):
        r"""Updates the layers of the network according to the PEPITA learning rule (https://arxiv.org/pdf/2201.11665.pdf)

        Args:
            x (torch.Tensor): the network input
            e (torch.Tensor): the error computed at the output after the first forward pass
            lr (float): the learning rate
            batch_size (int): the batch size
            first_block (bool, optional): see https://arxiv.org/pdf/2201.11665.pdf for reference (default is True)
            last_block (bool, optional): see https://arxiv.org/pdf/2201.11665.pdf for reference (default is True)
        """
        if first_block:
            hl_err = x + (e @ self.B.T)
        else:
            hl_err = x
        # TODO add activations for forward and modulated
        forward_activations = self.get_activations()
        modulated_forward = self.forward(hl_err)
        modulated_activations = self.get_activations()
        
        for l, layer in enumerate(self.layers):
            if (l == len(self.layers) - 1) and last_block:
                dwl = e.T @ (modulated_activations[l - 1] if l != 0 else x)
                layer[0].weight -= lr * dwl / batch_size
            else:
                dwl = (forward_activations[l] - modulated_activations[l]).T @ (modulated_activations[l - 1] if l != 0 else hl_err)
            layer[0].weight -= lr * dwl / batch_size
        
        self.reset_dropout_masks()
            
        return modulated_forward
            