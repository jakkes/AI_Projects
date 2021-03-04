from typing import Dict, Tuple
from abc import ABC, abstractmethod

import numpy as np


class Base(ABC):

    """Environment base class.

    An environment is a stateful environment upon which action may be executed. It has
    an internal state that is modified by the action and (potentially only partially)
    observable from the outside.

    States are given as `np.ndarray`s. Actions are, for now, only discrete and given by
    their action index. States are always returned with a corresponding action mask,
    indicating which actions are legal in the given state."""

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Resets the environment to a new initial state.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of state and boolean action mask.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """Executes an action in the environment.

        Args:
            action (int): Action index

        Returns:
            Tuple[np.ndarray, np.ndarray, float, bool, Dict]: Tuple of next state, next
            action mask, reward, terminal flag, and debugging dictionary.
        """
        raise NotImplementedError
