"""Utility methods and classes for neural networks defined in PyTorch."""

from math import sqrt

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, std_init, bias=True):
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

    def forward(self, x):
        if self.training:
            epsin = NoisyLinear.get_noise(self.in_features)
            epsout = NoisyLinear.get_noise(self.out_features)
            self.weps = epsout.ger(epsin)
            self.beps = self.get_noise(self.out_features)
            # self.weps.copy_(epsout.ger(epsin))
            # self.beps.copy_(self.get_noise(self.out_features))

            return super().forward(x) + F.linear(
                x, self.noise_weight * self.weps, self.noise_bias * self.beps
            )
        else:
            return super().forward(x)

    @staticmethod
    @torch.jit.script
    def get_noise(size: int) -> Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt_()
