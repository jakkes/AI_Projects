from typing import Tuple

import torch

from ._base import Base


class Uniform(Base):
    """Buffer from which samples are drawn uniformly."""

    def __init__(
        self,
        capacity: int,
        shapes: Tuple[Tuple[int, ...]],
        dtypes: Tuple[torch.dtype],
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            capacity (int): Capacity of the buffer.
            shapes (Tuple[Tuple[int, ...]]): Shapes of the data to store.
            dtypes (Tuple[torch.dtype]): Data types of the data to store.
            device (torch.device, optional): Device on which to store the data. Defaults
                to CPU.
        """
        self._data = tuple(
            torch.zeros((capacity, ) + shape, dtype=dtype, device=device)
            for shape, dtype in zip(shapes, dtypes)
        )
        self._capacity = capacity
        self._i = 0
        self._full = False

    def sample(self, n: int) -> Tuple[Tuple[torch.Tensor], torch.Tensor, torch.Tensor]:
        i = torch.randint(0, self.size, (n, ))
        return (
            tuple(x[i] for x in self._data),
            torch.ones(n) / self.size,
            i
        )

    def get_all(self) -> Tuple[Tuple[torch.Tensor], torch.Tensor, torch.Tensor]:
        return (
            tuple(x[:self.size] for x in self._data),
            torch.ones(self.size),
            torch.arange(self.size)
        )

    def update_weights(self, identifiers: torch.Tensor, weights: torch.Tensor):
        pass

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._capacity if self._full else self._i

    def add(self, data: Tuple[torch.Tensor], weights: torch.Tensor, batch: bool=True) -> torch.Tensor:
        if not batch:
            data = tuple(x.unsqueeze(0) for x in data)
        i = (self._i + torch.arange(data[0].shape[0])) % self._capacity
        for x, y in zip(data, self._data):
            y[i] = x

        self._i += i.shape[0]
        if self._i >= self._capacity:
            self._i %= self._capacity
            self._full = True
        return i if batch else i[0]
