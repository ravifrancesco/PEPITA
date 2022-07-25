import numpy as np

from loguru import logger

import torch
import torch.nn as nn

from scipy import spatial

from pepita.models.layers.ConsistentDropout import ConsistentDropout

# code adapted from https://github.com/avicooper1/CPSC490/blob/main/pepita.ipynb

@torch.no_grad()
def initialize_layer(layer, in_size, init='he_uniform'):
    r"""Initializes layer according to the input init mode

    Args:
        layer (torch.Tensor): layer to initialize
        in_size (int): layer input size
        init (str, optional): initialization mode (default is 'he_uniform')
    """
    if init.lower()=='he_uniform':
       layer_limit = np.sqrt(6.0 / in_size) 
       torch.nn.init.uniform_(layer.weight, a=-layer_limit, b=layer_limit)
    elif init.lower()=='he_normal':
        layer_limit = np.sqrt(2.0 / in_size)
        torch.nn.init.normal_(layer.weight, mean=0.0, std=layer_limit)
    else:
        logger.error(f'Initialization {init.lower()} not implemented yet')
        exit()

@torch.no_grad()
def generate_layer(in_size, out_size, p=0.1, final_layer=False, init='he_uniform'):
    r"""Helper function to generate a Fully Connected block

    Args:
        in_size (int): input size
        out_size (int): output size
        p (float, optional): dropout rate (default is 0.1)
        final_layer (bool, optional): if True, the layer is treated as final layer (default is False)
        init (str, optional): initialization mode (default is 'he_uniform')
    Returns:
        Fully connected block (nn.Sequential): fully connected block of the specified dimensions
    """
    w = nn.Linear(in_size, out_size, bias=False)
    initialize_layer(w, in_size, init)

    if final_layer:
        a = nn.Softmax(dim=1)
        return nn.Sequential(w, a)
    else:
        d = ConsistentDropout(p=p)
        a = nn.ReLU()
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

def generate_B(n_in, n_out, std, init='normal'):
    r"""Helper function to generate the feedback matrix

    Args:
        n_in (int): network input size
        n_out (int): network output size
        init (str, optional): initialization mode (default is 'uniform')
        B_mean_zero (bool, optional): if True, the distribution of the entries is centered around 0 (default is True)
        Bstd (float, optional): standard deviation of the entries of the matrix (default is 0.05)
    
    Returns:
        Tensor (torch.Tensor): the feedback matrix
    """
    if init.lower()=='normal':
        sd = np.sqrt(2.0 / n_in)
        B = torch.empty(n_in, n_out).normal_(mean=0,std=std)
    else:
        logger.error(f'B initialization \'{init.lower()}\' is not valid ')
        
    return B

class FCNetMirror(nn.Module):
    r"""Fully connected network

    Attributes:
        layers (nn.Module): layers of the network
        B (torch.Tensor): feedback matrix
    """
    @torch.no_grad()
    def __init__(self, layer_sizes, init='he_normal', B_init='normal', B_mean_zero=True, Bstd=0.05, p=0.1, final_layer=True, wmlr=0.1, wmwd=0.5):
        r"""
        Args:
            layer_sizes (list[int]): sizes of each layers
            init (str, optional): layer initialization mode (default is 'he_uniform')
            B_init (str, optional): B initialization mode (default is 'uniform')
            B_mean_zero (bool, optional): if True, the distribution of the entries is centered around 0 (default is True)
            Bstd (float, optional): standard deviation of the entries of the matrix (default is 0.05)
            p (float, optional): dropout rate (default is 0.1) 
            final_layer (bool, optional): if True, the last layer is treated as the last layer of the network (default is True)
        """
        super(FCNetMirror, self).__init__()

        self.layers_list = [generate_layer(in_size, out_size, p=p, init=init) for in_size, out_size in zip(layer_sizes, layer_sizes[1:-1])]
        self.layers_list.append(generate_layer(layer_sizes[-2], layer_sizes[-1], p=p, final_layer=final_layer, init=init))
        
        self.layers = nn.Sequential(*self.layers_list)
        
        self.weights = [layer[0].weight for layer in self.layers]
        
        self.activations = [None] * len(self.layers)
        
        for l, layer in enumerate(self.layers):
            layer.register_forward_hook(collect_activations(self, l))


        sd = np.sqrt(2.0 / layer_sizes[0]) * Bstd
        logger.info(f'Expected STD = {torch.std(torch.empty(layer_sizes[0], layer_sizes[-1]).normal_(mean=0,std=np.sqrt(2.0 / layer_sizes[0])) * Bstd)}')
            
        self.Bs = []
        self.Bs.append(generate_B(layer_sizes[0], layer_sizes[1], (sd / (layer_sizes[1]**(1./2.)))**(1./2.)))
        self.Bs.append(generate_B(layer_sizes[1], layer_sizes[-1], (sd / (layer_sizes[1]**(1./2.)))**(1./2.)))
        logger.info(f'Generated feedback matrix 1 with shape {self.Bs[0].shape}')
        logger.info(f'Generated feedback matrix 2 with shape {self.Bs[1].shape}')
        logger.info(f'Obtained STD = {torch.std(self.Bs[0] @ self.Bs[1])}')
        logger.info(f'Obtained Mean = {torch.mean(self.Bs[0] @ self.Bs[1])}')

        self.wmlr = wmlr
        self.wmwd = wmwd
        
    @torch.no_grad()
    def get_activations(self):
        r"""Returns the activations of the network
        
        Returns:
            activations (list[torch.Tensor]): the network activations
        """
        return [activations.clone() for activations in self.activations]

    def reset_dropout_masks(self):
        r"""Resets dropout masks
        """
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
    def modulated_forward(self, x, e, batch_size, noise_amplitude=0.1):
        r"""Updates the layers gradient according to the PEPITA learning rule (https://arxiv.org/pdf/2201.11665.pdf)

        Args:
            x (torch.Tensor): the network input
            e (torch.Tensor): the error computed at the output after the first forward pass
            batch_size (int): the batch size
           
        Returns:
            modulated_forward (torch.Tensor): modulated output
        """
        # Engaged mode
        B = self.Bs[0] @ self.Bs[1]
        
        hl_err = x + (e @ B.T)

        forward_activations = self.get_activations()
        modulated_forward = self.forward(hl_err)
        modulated_activations = self.get_activations()
        
        for l, layer in enumerate(self.layers):
            if (l == len(self.layers) - 1):
                dwl = e.T @ (modulated_activations[l - 1] if l != 0 else x)
            else:
                dwl = (forward_activations[l] - modulated_activations[l]).T @ (modulated_activations[l - 1] if l != 0 else hl_err)
            layer[0].weight.grad = dwl / batch_size
        
        # Mirror mode
        for l, layer in enumerate(self.layers):
            noise_x = noise_amplitude * (torch.randn(batch_size, self.weights[l].shape[1]) - 0.5)
            noise_y = layer(noise_x)
            # update the backward weight matrices using the equation 7 of the paper manuscript
            update = noise_x.T @ noise_y / batch_size
            self.Bs[l] = (1-self.wmwd) * self.Bs[l] + self.wmlr * update
        # Dropout masks reset
        
        self.reset_dropout_masks()
            
        return modulated_forward
    
    def get_B(self):
        return self.Bs[0] @ self.Bs[1]

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

    @torch.no_grad()
    def compute_angle(self):
        r"""Returns angle between feedforward matrix and feedback matrix TODO adapt other functions to support multiple angles
        """
        cos0 = 1-spatial.distance.cosine(self.weights[0].flatten(), self.Bs[0].flatten())
        ang0 = np.arccos(cos0)*180/np.pi
        cos1 = 1-spatial.distance.cosine(self.weights[1].flatten(), self.Bs[1].flatten())
        ang1 = np.arccos(cos1)*180/np.pi

        return {'w0': ang0, 'w1': ang1}

            