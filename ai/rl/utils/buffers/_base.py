import abc
from typing import Tuple

import torch


class Base(abc.ABC):
    """Base buffer class. A buffer is a collection of experiences."""

    @abc.abstractmethod
    def get_all(self) -> Tuple[Tuple[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Collects and returns all data found in the buffer.

        Returns:
            Tuple[Tuple[torch.Tensor], torch.Tensor, torch.Tensor]: Tuple of (data,
                weights, identifier).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Size of the buffer."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capacity(self) -> int:
        """Capacity of the buffer."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, n: int) -> Tuple[Tuple[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Collects samples from the buffer.

        Args:
            n (int): Number of samples to collect.

        Returns:
            Tuple[Tuple[torch.Tensor], torch.Tensor, torch.Tensor]: Tuple of (data,
                sample_probabilities, identifier).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self, identifiers: torch.Tensor, weights: torch.Tensor):
        """Updates the weights of the given samples.

        Args:
            identifiers (torch.Tensor): Identifiers of the samples whose weights shall
                be updated.
            weights (torch.Tensor): New weights.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def add(self, data: Tuple[torch.Tensor], weights: torch.Tensor, batch: bool=True) -> torch.Tensor:
        """Adds new data to the buffer.

        Args:
            data (Tuple[torch.Tensor]): Data to be added.
            weights (torch.Tensor): Weights of the new data.
            batch (optional, bool): If `True`, then the first dimension of each tensor
                is treated as a batch dimension, allowing batch inserts. Defaults to
                `True`.

        Returns:
            torch.Tensor: Identifier given to the new data.
        """
        raise NotImplementedError
