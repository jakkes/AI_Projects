import math
from typing import List

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.lin_k = nn.Linear(d_model, d_k, bias=False)
        self.lin_q = nn.Linear(d_model, d_k, bias=False)
        self.lin_v = nn.Linear(d_model, d_v, bias=False)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, k, q, v):
        return torch.softmax(
            self.lin_q(q).matmul(self.lin_k(k).transpose(-1, -2)).div_(self.sqrt_d_k),
            dim=-1,
        ).matmul(self.lin_v(v))


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.attentions = nn.ModuleList(
            [Attention(d_model, d_k, d_v) for _ in range(h)]
        )
        self.lin = nn.Linear(h * d_v, d_model, bias=False)

    def forward(self, k, q, v):
        return self.lin(
            torch.cat([attention(k, q, v) for attention in self.attentions], dim=-1)
        )


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
