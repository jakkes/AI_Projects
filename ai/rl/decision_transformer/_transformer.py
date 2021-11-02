import math

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.lin_k = nn.Linear(d_model, d_k, bias=False)
        self.lin_q = nn.Linear(d_model, d_k, bias=False)
        self.lin_v = nn.Linear(d_model, d_v, bias=False)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, x, mask=None):
        if mask is None:
            return torch.softmax(
                self.lin_q(x)
                .matmul(self.lin_k(x).transpose(-1, -2))
                .div_(self.sqrt_d_k),
                dim=-1,
            ).matmul(self.lin_v(x))
        else:
            imask = ~mask
            presoftmax = (
                self.lin_q(x)
                .matmul(self.lin_k(x).transpose(-1, -2))
                .div_(self.sqrt_d_k)
            )
            presoftmax[imask] = -math.inf
            softmax = torch.softmax(presoftmax, dim=-1)
            softmax[torch.where(torch.all(imask, dim=-1))] = 0
            return softmax.matmul(self.lin_v(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.attentions = nn.ModuleList(
            [Attention(d_model, d_k, d_v) for _ in range(h)]
        )
        self.lin = nn.Linear(h * d_v, d_model, bias=False)

    def forward(self, x, mask=None):
        return self.lin(
            torch.cat(
                [attention(x, mask=mask) for attention in self.attentions], dim=-1
            )
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, h: int, d_model: int, d_k: int, d_v: int, fc_hidden_layer_size: int
    ):
        super().__init__()
        self.mha = MultiHeadAttention(h, d_model, d_k, d_v)
        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_layer_size, d_model),
        )
        self.layernorm1 = nn.LayerNorm((d_model,))
        self.layernorm2 = nn.LayerNorm((d_model,))

    def forward(self, x, mask=None):
        x = self.layernorm1(x + self.mha(x, mask=mask))
        return self.layernorm2(x + self.fc(x))


class TransformerEncoder(nn.Module):
    """A transformer encoder module. This module performs the sequence encoding
    operation of a full transformer module."""

    def __init__(
        self,
        N: int,
        h: int,
        d_model: int,
        d_k: int,
        d_v: int,
        fc_hidden_layer_size: int,
    ):
        """
        Args:
            N (int): Number of encoder layers to be used.
            h (int): Number of heads in the multi head attention.
            d_model (int): Size of encoded vectors passed to this module.
            d_k (int): Size of key vectors used in the attention modules.
            d_v (int): Size of value vectors used in the attention modules.
            fc_hidden_layer_size (int): Size of the hidden layer processing the output
                of each multi head attention.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(h, d_model, d_k, d_v, fc_hidden_layer_size)
                for _ in range(N)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
