import numpy as np

from scipy import spatial

from loguru import logger

import torch
import torch.nn as nn

from pepita.models.layers.RandomFeedback import RandomFeedback

from pepita.models.FCnet import (
    initialize_layer,
    generate_layer,
    collect_activations,
    FCNet
)


@torch.no_grad()
def generate_conv_layer(in_size, out_size, stride=1, padding=0, final_layer=False, init="he_uniform"):
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
    w = nn.Conv2d(
        in_size,
        out_size,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False
    )
    initialize_layer(w, in_size, init)

    if final_layer:
        a = nn.Softmax(dim=1)
    else:
        a = nn.ReLU()
    return nn.Sequential(w, a)


class ConvNet(FCNet):
    r"""Fully connected network

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
        fc_dropout_p=0.,
        init="he_uniform",
        B_init="uniform",
        B_mean_zero=True,
        Bstd=0.05,
        final_layer=True,
    ):
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
        nn.Module.__init__(self)

        # Generating network
        # Convolutional stack
        self.layers_list = [
            generate_conv_layer(in_size, out_size, init=init)
            for in_size, out_size in zip(conv_channels, conv_channels[1:])
        ]

        # Final layers
        self.layers_list.append(nn.Flatten())
        if fc_dropout_p:
            self.layers_list.append(nn.Dropout(p=fc_dropout_p))
        self.layers_list.append(generate_layer(
            fc_layer_size, n_classes, p=False, final_layer=final_layer, init=init
        ))

        self.layers = nn.Sequential(*self.layers_list)
        self.weights = [layer[0].weight for layer in self.layers]
        self.activations = [None] * len(self.layers)
 
        for l, layer in enumerate(self.layers):
            layer.register_forward_hook(collect_activations(self, l))

        # Generating B
        layer_sizes = (conv_channels[0], n_classes)
        input_shape = (conv_channels[0], *img_shape)
        self.Bs = RandomFeedback(layer_sizes, init=B_init, B_mean_zero=B_mean_zero, Bstd=Bstd, img_shape=input_shape)

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

        if len(self.layers) == 1:  # limit case of one layer, just delta rule
            dwl = e.T @ x
            self.layers[0][0].weight.grad = dwl / batch_size

        if output_mode == "mixed":
            for l, layer in enumerate(self.layers):
                if l == len(self.layers) - 1:
                    dwl = y.T @ forward_activations[l - 1] - target.T @ modulated_activations[l - 1]
                elif l == 0:
                    dwl = forward_activations[l].T @ x - modulated_activations[l].T @ hl_err
                else:
                    dwl = (
                        forward_activations[l].T @ forward_activations[l - 1] -
                        modulated_activations[l].T @ modulated_activations[l - 1]
                    )
                layer[0].weight.grad = dwl / batch_size

        else:  # original Pepita
            for l, layer in enumerate(self.layers):
                if l == len(self.layers) - 1:
                    dwl = e.T @ output_activations[l - 1]
                else:
                    dwl = (forward_activations[l] - modulated_activations[l]).T @ (
                        output_activations[l - 1] if l != 0 else hl_err
                    )
                layer[0].weight.grad = dwl / batch_size

        self.reset_dropout_masks()

        return modulated_forward

    @torch.no_grad()
    def mirror_weights(self, batch_size, noise_amplitude=0.1):
        r"""Perform weight mirroring

        Args:
            batch_size (int): batch size
            noise_amplitude (float, optional): noise amplitude (default is 0.1)
            wmlr (float, optional): learning rate (default is 0.01)
            wmwd (float, optional): weigth decay (default is 0.0001)
        """

        if len(self.weights) != len(self.get_Bs()):
            logger.error("The B initialization is not valid for performing mirroring")
            exit()

        for l, layer in enumerate(self.layers):
            noise_x = noise_amplitude * (
                torch.randn(batch_size, self.weights[l].shape[1])
            )
            noise_x -= torch.mean(noise_x).item()
            device = next(layer.parameters()).device
            noise_x = noise_x.to(device)
            noise_y = layer(noise_x)
            # update the backward weight matrices using the equation 7 of the paper manuscript
            update = noise_x.T @ noise_y / batch_size
            self.get_Bs()[l].grad = -update.cpu()

        self.reset_dropout_masks()

    
    @torch.no_grad()
    def normalize_B(self):
        r"""Normalizes Bs to keep std constant"""
        self.Bs.normalize_B()

    @torch.no_grad()
    def get_B(self):
        r"""Returns B (from output to input)"""
        return self.Bs.get_B()

    @torch.no_grad()
    def get_Bs(self):
        r"""Returns Bs"""
        return self.Bs.get_Bs()

    @torch.no_grad()
    def get_tot_weights(self):
        r"""Returns the total weight matrix (input to output matrix)"""
        weights = self.weights[0]
        for w in self.weights[1:]:
            weights = w @ weights
        return weights

    @torch.no_grad()
    def get_weights_norm(self):
        r"""Returns dict with norms of weight matrixes"""
        d = {}
        for i, w in enumerate(self.weights):
            d[f"layer{i}"] = torch.linalg.norm(w)
        return d

    @torch.no_grad()
    def get_B_norm(self):
        r"""Returns dict with norms of weight matrixes"""
        d = {}
        for i, b in enumerate(self.get_Bs()):
            d[f"layer{i}"] = torch.linalg.norm(b)
        return d

    @torch.no_grad()
    def compute_angle(self):
        r"""Returns alignment dictionary between feedforward and feedback matrices"""
        cost = 1 - spatial.distance.cosine(
            self.get_tot_weights().T.cpu().flatten(), self.get_B().cpu().flatten()
        )
        angt = np.arccos(cost) * 180 / np.pi
        angles = {"al_total": angt}
        if len(self.get_Bs()) > 1:
            for l, b in enumerate(self.get_Bs()):
                cos = 1 - spatial.distance.cosine(
                    self.weights[l].T.cpu().flatten(), b.cpu().flatten()
                )
                ang = np.arccos(cos) * 180 / np.pi
                angles[f"al_layer{l}"] = ang
        return angles