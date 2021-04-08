from typing import Tuple, Sequence, Optional, Union

from numpy import ndarray

import torch
from torch import Tensor


class NStepRewardCollector:
    """Utility object for collecting n-step rewards."""

    def __init__(
        self,
        n_step: int,
        discount_factor: float,
        state_data_shapes: Sequence[Tuple[int, ...]],
        state_data_dtypes: Sequence[torch.dtype],
    ):
        """
        Args:
            n_step (int): N-step to apply.
            discount_factor (float): Discount factor.
            state_data_shapes (Sequence[Tuple[int, ...]]): Sequence of shapes that need
                to be stored at each state. These tensors are then paired with the
                correct state and next states.
            state_data_dtypes (Sequence[torch.dtype]): Sequence of data types that need
                to be stored at each state.
        """
        self._n_step = n_step
        self._discount = discount_factor
        self._state_data_buffer = tuple(
            torch.empty((n_step,) + shape, dtype=dtype)
            for shape, dtype in zip(state_data_shapes, state_data_dtypes)
        )
        self._rewards = torch.zeros(n_step, n_step, dtype=torch.float32)
        self._index_vector = torch.arange(n_step)
        self._discount_vector = torch.ones(n_step).pow_(self._index_vector)
        self._i = 0
        self._looped = False

    def step(
        self,
        reward: float,
        terminal: bool,
        state_data: Sequence[Union[Tensor, ndarray]],
    ) -> Optional[Tuple[Sequence[Tensor], Tensor, Sequence[Tensor]]]:
        """Observes one state transition

        Args:
            reward (float): Reward observed in the 1-step transition.
            terminal (bool): If the state transition resulted in a terminal state.
            state_data (Sequence[Union[Tensor, ndarray]]): State data of the state that
                was transitioned _from_.

        Returns:
            Optional[Tuple[Sequence[Tensor], Tensor, Sequence[Tensor]]]: If available:
            Tuple of state data, rewards, next state data. Otherwise, None.
        """

        index_vector = (self._index_vector + self._i) % self._n_step
        self._rewards[index_vector, self._index_vector].fill_(reward)

