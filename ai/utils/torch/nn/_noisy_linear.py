"""Utility methods and classes for neural networks defined in PyTorch."""

from math import sqrt

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class NoisyLinear(nn.Linear):
    """Implementation of a noisy (linear) network.

    Noisy networks add random noise to the activation, scaled by learnable weights."""

    def __init__(
        self, in_features: int, out_features: int, std_init: float, bias: bool = True
    ):
        """
        Args:
            in_features (int): Number of in features.
            out_features (int): Number of out features.
            std_init (float): Initial standard deviation of the noise.
            bias (bool, optional): If True, uses a bias a term in the linear
                transformation. Defaults to True.
        """
        super().__init__(in_features, out_features, bias)

        self.noise_weight = nn.Parameter(Tensor(out_features, in_features))
        self.noise_weight.data.fill_(std_init / sqrt(in_features))

        if bias:
            self.noise_bias = nn.Parameter(Tensor(out_features))
            self.noise_bias.data.fill_(std_init / sqrt(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("weps", torch.zeros(out_features, in_features))
        self.register_buffer("beps", torch.zeros(out_features))

        self._device: torch.device = None

    def forward(self, x):
        """"""
        if self.training:
            epsin = self.get_noise(self.in_features, self.noise_weight.dtype, self.noise_weight.device)
            epsout = self.get_noise(self.out_features, self.noise_weight.dtype, self.noise_weight.device)
            self.weps = epsout.ger(epsin)
            self.beps = self.get_noise(self.out_features, self.noise_weight.dtype, self.noise_weight.device)

            return super().forward(x) + F.linear(
                x, self.noise_weight * self.weps, self.noise_bias * self.beps
            )
        else:
            return super().forward(x)

    @staticmethod
    @torch.jit.script
    def get_noise(size: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        x = torch.randn(size, dtype=dtype, device=device)
        return x.sign() * x.abs().sqrt_()
