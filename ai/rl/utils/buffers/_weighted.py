from math import log2
from typing import Tuple, Union

import torch
from torch import Tensor

from ._base import Base


class Weighted(Base):
    """Buffer where samples are drawn according to their sample weight."""

    def __init__(
        self,
        capacity: int,
        priority_exponent: float,
        shapes: Tuple[Tuple[int, ...]],
        dtypes: Tuple[torch.dtype],
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            capacity (int): Capacity of the buffer.
            priority_exponent (float): Exponent controlling the strictness of the
                distribution. Larger value implies samples with larger weights are given
                higher priority. Zero implies uniform sampling. One implies proportional
                to sample weight.
            shapes (Tuple[Tuple[int, ...]]): Shapes of the data to store.
            dtypes (Tuple[torch.dtype]): Data types of the data to store.
            device (torch.device, optional): Device on which to store the data. Defaults
                to CPU.
        """
        if log2(capacity) % 1 != 0:
            raise ValueError(
                "Capacity must be power of two, when using a weighted buffer"
            )

        self._data = tuple(
            torch.zeros((capacity,) + shape, dtype=dtype, device=device)
            for shape, dtype in zip(shapes, dtypes)
        )
        self._weights = torch.zeros(capacity * 2 - 1, device=device)
        self._wi = capacity - 1
        self._depth = int(log2(capacity))

        self._capacity = capacity
        self._i = 0
        self._full = False
        self._alpha = priority_exponent
        self._device = device

    @property
    def size(self) -> int:
        return self._capacity if self._full else self._i

    @property
    def capacity(self) -> int:
        return self._capacity

    def get_all(self) -> Tuple[Tuple[Tensor], Tensor, Tensor]:
        return (
            tuple(x[: self.size] for x in self._data),
            self._weights[self._wi : self._wi + self.size],
            torch.arange(self.size),
        )

    def add(self, data: Tuple[Tensor, ...], weights: Tensor, batch: bool=True) -> Tensor:
        if not batch:
            data = tuple(x.unsqueeze(0) for x in data)
        i = (self._i + torch.arange(data[0].shape[0])) % self._capacity
        for x, y in zip(data, self._data):
            y[i] = x
        self._set_weights(weights, i)

        self._i += i.shape[0]
        if self._i >= self._capacity:
            self._i %= self._capacity
            self._full = True
        return i if batch else i[0]

    def sample(self, n: int) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor]:
        #################################################
        #### Sample in bins for more stable sampling ####
        bin_starts = torch.linspace(0, self._weights[0].item(), steps=n + 1)[
            :-1
        ]  # Exclude endpoint
        bin_spacing = bin_starts[1] - bin_starts[0]
        random_samples = torch.rand(n) * bin_spacing
        w = random_samples + bin_starts
        #### End bin sampling ####
        ##########################

        i = self._retrieve_indices(w)
        return (
            tuple(x[i] for x in self._data),
            self._weights[self._wi + i] / self._weights[0],
            i
        )

    def update_weights(self, identifiers: torch.Tensor, weights: torch.Tensor):
        self._set_weights(weights, identifiers)

    def _set_weight(self, weight: float, i: int):
        i = torch.tensor([i])
        weight = torch.tensor(weight)
        self._set_weights(weight, i)

    def _set_weights(self, weights: Union[Tensor, float], i: Tensor):

        ######################################
        ###### Remove duplicate entries ######
        weights = torch.as_tensor(weights).pow_(self._alpha)
        _, argi = torch.unique(i, return_inverse=True)
        i = i[argi]
        if len(weights.shape) > 0:
            weights = weights[argi]
        ###### Done removing duplicate entries ######
        #############################################

        n = i.shape[0]
        wi = self._wi + i
        dw = weights - self._weights[wi]

        w_update_i = torch.empty(
            (n, self._depth + 1), dtype=torch.long, device=self._device
        )
        w_update_i[:, 0] = wi
        for d in range(1, self._depth + 1):
            w_update_i[:, d] = (w_update_i[:, d - 1] - 1) // 2

        for j in range(n):
            self._weights[w_update_i[j]] += dw[j]

    def _retrieve_indices(self, w: Tensor, w_inplace=True):
        if not w_inplace:
            w = w.clone()

        i = torch.zeros_like(w, dtype=torch.long)
        while i[0] * 2 + 1 < self._weights.shape[0]:
            left = 2 * i + 1
            right = 2 * i + 2

            go_right = w > self._weights[left]
            w[go_right] -= self._weights[left[go_right]]

            i[go_right] = right[go_right]
            i[~go_right] = left[~go_right]
        return i - self._wi
