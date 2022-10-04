import numpy as np

from scipy import spatial

from loguru import logger

import torch
import torch.nn as nn

from pepita.models.layers.ConsistentDropout import ConsistentDropout

# code adapted from https://github.com/avicooper1/CPSC490/blob/main/pepita.ipynb


@torch.no_grad()
def initialize_layer(layer, in_size, init="he_uniform"):
    r"""Initializes layer according to the input init mode

    Args:
        layer (torch.Tensor): layer to initialize
        in_size (int): layer input size
        init (str, optional): initialization mode (default is 'he_uniform')
    """
    if init.lower() == "he_uniform":
        layer_limit = np.sqrt(6.0 / in_size)
        torch.nn.init.uniform_(layer.weight, a=-layer_limit, b=layer_limit)
    elif init.lower() == "he_normal":
        layer_limit = np.sqrt(2.0 / in_size)
        torch.nn.init.normal_(layer.weight, mean=0.0, std=layer_limit)
    else:
        logger.error(f"Initialization {init.lower()} not implemented yet")
        exit()


@torch.no_grad()
def generate_layer(in_size, out_size, p=0.1, final_layer=False, init="he_uniform"):
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


def generate_B(layer_sizes, init="uniform", B_mean_zero=True, Bstd=0.05):
    r"""Helper function to generate the feedback matrix. If normal initialization is chosen, multiple matrices are generated
    (one per forward matrix).

    Args:
        layer_sizes (list(int)): size of layers of the networt
        init (str, optional): initialization mode (default is 'uniform')
        B_mean_zero (bool, optional): if True, the distribution of the entries is centered around 0 (default is True)
        Bstd (float, optional): standard deviation of the entries of the matrix (default is 0.05)

    Returns:
        Bs (list(torch.Tensor)): the feedback matrix (or matrices)
    """
    Bs = []
    if init.lower() == "uniform":
        sd = np.sqrt(6.0 / layer_sizes[-1])
        if B_mean_zero:
            Bs.append(
                (torch.rand(layer_sizes[0], layer_sizes[-1]) * 2 * sd - sd) * Bstd
            )
        else:
            Bs.append((torch.rand(layer_sizes[0], layer_sizes[-1]) * sd) * Bstd)
        logger.info(f"Generated feedback matrix with shape {Bs[0].shape}")
    elif init.lower() == "normal":
        sd = np.sqrt(2.0 / layer_sizes[0]) * Bstd
        n = len(layer_sizes) - 1
        el = np.prod(layer_sizes[1:-1])
        for i, (size0, size1) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            Bs.append(
                torch.empty(size0, size1).normal_(
                    mean=0, std=(sd / (el ** (1.0 / 2.0))) ** (1.0 / n)
                )
            )
            logger.info(f"Generated feedback matrix {i} with shape {Bs[i].shape}")
    else:
        logger.error(f"B initialization '{init.lower()}' is not valid ")

    return Bs


class FCNet(nn.Module):
    r"""Fully connected network

    Attributes:
        layers (nn.Module): layers of the network
        Bs (list(torch.Tensor)): feedback matrices
        B (torch.Tensor): feedback matrix
    """

    @torch.no_grad()
    def __init__(
        self,
        layer_sizes,
        init="he_uniform",
        B_init="uniform",
        B_mean_zero=True,
        Bstd=0.05,
        p=0.1,
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
        super(FCNet, self).__init__()

        # Generating network
        self.layers_list = [
            generate_layer(in_size, out_size, p=p, init=init)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:-1])
        ]
        self.layers_list.append(
            generate_layer(
                layer_sizes[-2],
                layer_sizes[-1],
                p=p,
                final_layer=final_layer,
                init=init,
            )
        )

        self.layers = nn.Sequential(*self.layers_list)

        self.weights = [layer[0].weight for layer in self.layers]

        self.activations = [None] * len(self.layers)

        for l, layer in enumerate(self.layers):
            layer.register_forward_hook(collect_activations(self, l))

        # Generating B
        self.Bs = generate_B(
            layer_sizes, init=B_init, B_mean_zero=B_mean_zero, Bstd=Bstd
        )
        if (len(self.Bs)) > 1:
            logger.info(
                f"Expected STD = {torch.std(torch.empty(layer_sizes[0], layer_sizes[-1]).normal_(mean=0,std=np.sqrt(2.0 / layer_sizes[0])) * Bstd)}"
            )
            logger.info(f"Obtained STD = {torch.std(self.get_B())}")
            logger.info(f"Obtained Mean = {torch.mean(self.get_B())}")

    @torch.no_grad()
    def get_activations(self):
        r"""Returns the activations of the network

        Returns:
            activations (list[torch.Tensor]): the network activations
        """
        return [activations.clone() for activations in self.activations]

    def reset_dropout_masks(self):
        r"""Resets dropout masks"""
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
    def modulated_forward(self, x, y, target, batch_size, output_mode="modulated"):
        r"""Updates the layers gradient according to the PEPITA learning rule (https://arxiv.org/pdf/2201.11665.pdf)

        Args:
            x (torch.Tensor): the network input
            y (torch.Tensor): the output of the first forward pass
            target (torch.Tensor): the one-hot encoded target
            batch_size (int): the batch size
            output_mode (str): One of "forward", "modulated", "mixed". Indicates whether \
                to use the first or second forward pass for the output term in the learning \
                rule, or split the terms and use one each (fully Hebbian - fully anti-Hebbian).

        Returns:
            modulated_forward (torch.Tensor): modulated output
        """
        assert output_mode in ("modulated", "forward", "mixed"), "Output mode not recognized"

        inp_err = x - (e @ self.get_B().T)
        forward_activations = self.get_activations()
        modulated_forward = self.forward(inp_err)
        modulated_activations = self.get_activations()

        e = y - target  # TODO it would make more sense to get gt in input and compute one hot here.
        output_activations = forward_activations if output_mode == "forward" else modulated_activations
        hl_err = x if output_mode == "forward" else inp_err

        print(e)

        if len(self.layers) == 1:  # limit case of one layer, just delta rule
            dwl = e.T @ x
            self.layers[0][0].weight.grad = dwl / batch_size

        # splits dwl in two terms, one Hebbian, one anti-Hebbian
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

        if len(self.weights) != len(self.Bs):
            logger.error("The B initialization is not valid for performing mirroring")
            exit()

        for l, layer in enumerate(self.layers):
            noise_x = noise_amplitude * (
                torch.randn(batch_size, self.weights[l].shape[1])
            )
            noise_x -= torch.mean(noise_x).item()
            noise_y = layer(noise_x)
            # update the backward weight matrices using the equation 7 of the paper manuscript
            update = noise_x.T @ noise_y / batch_size
            if l == 0:
                self.Bs[l].grad = update
            else:
                self.Bs[l].grad = -update

        self.reset_dropout_masks()

    @torch.no_grad()
    def get_B(self):
        r"""Returns B (from output to input)"""
        B = self.Bs[0]
        for b in self.Bs[1:]:
            B = B @ b
        return B

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
    def compute_angle(self):
        r"""Returns alignment dictionary between feedforward and feedback matrices"""
        cost = 1 - spatial.distance.cosine(
            self.get_tot_weights().T.flatten(), self.get_B().flatten()
        )
        angt = np.arccos(cost) * 180 / np.pi
        angles = {"al_total": angt}
        for l, b in enumerate(self.Bs):
            cos = 1 - spatial.distance.cosine(
                self.weights[l].T.flatten(), self.Bs[l].flatten()
            )
            ang = np.arccos(cos) * 180 / np.pi
            angles[f"al_layer{l}"] = ang
        return angles
