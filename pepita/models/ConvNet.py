import numpy as np

from scipy import spatial

from loguru import logger

import torch
import torch.nn as nn

from pepita.models.layers.RandomFeedback import RandomFeedback
from pepita.models.layers.ConsistentDropout import ConsistentDropoutNd

from pepita.models.FCnet import (
    initialize_layer,
    generate_layer,
    collect_activations,
    FCNet
)


@torch.no_grad()
def generate_conv_layer(
        in_size, out_size, p=0.0,
        kernel=3, stride=1, padding=0,
        init="he_uniform"):
    r"""Helper function to generate a Fully Connected block

    Args:
        in_size (int): input size
        out_size (int): output size
        p (float, optional): dropout rate (default is 0.0)
        kernel, stride, padding: as in torch Conv2d implementation
        init (str, optional): initialization mode (default is 'he_uniform')
    Returns:
        Convolutional block (nn.Sequential): fully connected block of the specified dimensions
    """
    w = nn.Conv2d(in_size, out_size, kernel_size=kernel,
            stride=stride, padding=padding, bias=False)
    u = nn.Unfold(kernel_size=(kernel, kernel), stride=(stride, stride), padding=(padding, padding))
    initialize_layer(w, in_size, init)
    a = nn.ReLU()

    if p:
        dropout = ConsistentDropoutNd(p=p)
        return nn.Sequential(w, a, dropout), u
    else:
        return nn.Sequential(w, a), u


class ConvNet(FCNet):
    r"""A convolutional network model for PEPITA.
    The number of convolutional layers is arbitrary. The current implementation has a Dropout and a Flatten
    operation after the last convolution, and then an output Linear layer.
    Currently, kernel size is hardcoded to 3 and stride to 2.

    Attributes:
        layers (nn.Module): layers of the network
        Bs (list(torch.Tensor)): feedback matrices
        B (torch.Tensor): feedback matrix
    """

    @torch.no_grad()
    def __init__(
        self,
        conv_channels,
        fc_layer_size,
        n_classes,
        img_shape,
        dropout_p=0.,
        init="he_uniform",
        B_init="uniform",
        B_mean_zero=True,
        Bstd=0.05,
        final_layer=True,
    ):
        r"""
        Args:
            conv_channels (list[int]): number of channels for each convolutional layer. Determines no. of conv layers
            fc_layer_size (int): number of units in the fc layer after flattening (determined by input size and last conv size)
            n_classes (int): size of output
            img_shape (tuple[int]): height and width of the input
            dropout_p (float): dropout probability. Dropout is applied channel-wise after the last convolution (before flatten.)
            init (str, optional): layer initialization mode (default is 'he_uniform')
            B_init (str, optional): B initialization mode (default is 'uniform')
            B_mean_zero (bool, optional): if True, the distribution of the entries is centered around 0 (default is True)
            Bstd (float, optional): standard deviation of the entries of the matrix (default is 0.05)
            p (float, optional): dropout rate (default is 0.1)
            final_layer (bool, optional): if True, the last layer is treated as the last layer of the network (default is True)
        """
        nn.Module.__init__(self)

        # Generating network
        # Convolutional stack
        layers_list, self.unfold_objects = [], []
        for i, (in_size, out_size) in enumerate(zip(conv_channels, conv_channels[1:])):
            p = dropout_p if i == len(conv_channels) - 2 else 0.0  # LAST element has dropout
            conv, unfold = generate_conv_layer(
                in_size, out_size, p=p, kernel=3, init=init, stride=2, padding=1)
            layers_list.append(conv)
            self.unfold_objects.append(unfold)
        self.conv_layers = nn.Sequential(*layers_list)
        self.weights = [layer[0].weight for layer in self.conv_layers]

        # Final layers
        self.linear = generate_layer(
            fc_layer_size, n_classes, p=False, final_layer=final_layer, init=init)
        self.weights.append(self.linear[0].weight)
        
        # Hooks for recording activity
        self.activations = [None] * (len(self.conv_layers) + 1) # +1 is the linear 
        for l, layer in enumerate(self.conv_layers):
            layer.register_forward_hook(collect_activations(self, l))
        self.linear.register_forward_hook(collect_activations(self, l+1))

        # Generating B
        layer_sizes = (conv_channels[0]*img_shape[0]*img_shape[1], n_classes)  # TODO this could be done better
        input_shape = (conv_channels[0], *img_shape)
        self.Bs = RandomFeedback(
            layer_sizes, init=B_init, B_mean_zero=B_mean_zero,
            Bstd=Bstd, input_shape=input_shape)

    @torch.no_grad()
    def forward(self, x):
        r"""Computes the forward pass and returns the output

        Args:
            x (torch.Tensor): the input

        Returns:
            output (torch.Tensor): the network output
        """
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)

    @torch.no_grad()
    def modulated_forward(self, x, y, target, batch_size, output_mode="modulated"):
        r"""Updates the layers gradient according to the PEPITA learning rule (https://arxiv.org/pdf/2201.11665.pdf)

        Args:
            x (torch.Tensor): the network input
            e (torch.Tensor): the error computed at the output after the first forward pass
            batch_size (int): the batch size
            output_mode (str): One of "forward", "modulated", "mixed". Indicates whether \
                to use the first or second forward pass for the output term in the learning \
                rule, or split the terms and use one each (fully Hebbian - fully anti-Hebbian).

        Returns:
            modulated_forward (torch.Tensor): modulated output
        """

        assert output_mode in ("modulated", "forward", "mixed"), "Output mode not recognized"

        e = y - target
        inp_err = x - self.Bs(e)

        forward_activations = self.get_activations()
        modulated_forward = self.forward(inp_err)
        modulated_activations = self.get_activations()

        output_activations = forward_activations if output_mode == "forward" else modulated_activations
        hl_err = x if output_mode == "forward" else inp_err

        if output_mode == "mixed":
            raise NotImplementedError()
        else:  # original Pepita
            # for convolutional layers
            for l, layer in enumerate(self.conv_layers):
                input_term = output_activations[l - 1] if l != 0 else hl_err
                output_term = (forward_activations[l] - modulated_activations[l])
                unfolded_in = self.unfold_objects[l](input_term)
                k_h, k_w = self.unfold_objects[l].kernel_size

                batchsize, ch_in, h_in, w_in = input_term.shape
                batchsize, ch_out, h_out, w_out = output_term.shape
                unfolded_in = unfolded_in.reshape(batchsize, ch_in, k_h, k_w, h_out, w_out)

                dwl = torch.einsum("bcijhw, bkhw -> kcij", unfolded_in, output_term)
                layer[0].weight.grad = dwl / batch_size

            # for the linear layer
            input_term = output_activations[-2]  # check
            input_term = torch.flatten(input_term, start_dim=1)
            dwl = e.T @ input_term
            self.linear[0].weight.grad = dwl / batch_size

        self.reset_dropout_masks()

        return modulated_forward

    @torch.no_grad()
    def compute_angle(self):
        """Fake placeholder function."""
        angles = {"al_total": 0.}
        return angles

    # @torch.no_grad()
    # def mirror_weights(self, batch_size, noise_amplitude=0.1):
    #     r"""Perform weight mirroring

    #     Args:
    #         batch_size (int): batch size
    #         noise_amplitude (float, optional): noise amplitude (default is 0.1)
    #         wmlr (float, optional): learning rate (default is 0.01)
    #         wmwd (float, optional): weigth decay (default is 0.0001)
    #     """

    #     if len(self.weights) != len(self.get_Bs()):
    #         logger.error("The B initialization is not valid for performing mirroring")
    #         exit()

    #     for l, layer in enumerate(self.layers):
    #         noise_x = noise_amplitude * (
    #             torch.randn(batch_size, self.weights[l].shape[1])
    #         )
    #         noise_x -= torch.mean(noise_x).item()
    #         device = next(layer.parameters()).device
    #         noise_x = noise_x.to(device)
    #         noise_y = layer(noise_x)
    #         # update the backward weight matrices using the equation 7 of the paper manuscript
    #         update = noise_x.T @ noise_y / batch_size
    #         self.get_Bs()[l].grad = -update.cpu()

    #     self.reset_dropout_masks()

    # @torch.no_grad()
    # def normalize_B(self):
    #     r"""Normalizes Bs to keep std constant"""
    #     self.Bs.normalize_B()

    # @torch.no_grad()
    # def get_B(self):
    #     r"""Returns B (from output to input)"""
    #     return self.Bs.get_B()

    # @torch.no_grad()
    # def get_Bs(self):
    #     r"""Returns Bs"""
    #     return self.Bs.get_Bs()

    # @torch.no_grad()
    # def get_tot_weights(self):
    #     r"""Returns the total weight matrix (input to output matrix)"""
    #     weights = self.weights[0]
    #     for w in self.weights[1:]:
    #         weights = w @ weights
    #     return weights

    # @torch.no_grad()
    # def get_B_norm(self):
    #     r"""Returns dict with norms of weight matrixes"""
    #     d = {}
    #     for i, b in enumerate(self.get_Bs()):
    #         d[f"layer{i}"] = torch.linalg.norm(b)
    #     return d

