from typing import Dict, Tuple
from abc import ABC, abstractmethod

import numpy as np
from . import action_spaces


class Base(ABC):
    """Environment base class.

    An environment is a stateful environment upon which action may be executed. It has
    an internal state that is modified by the action and (potentially only partially)
    observable from the outside."""

    @property
    @abstractmethod
    def action_space(self) -> action_spaces.Base:
        """The action space instance used by the environment instance."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Resets the environment to a new initial state.

        Returns:
            np.ndarray: Initial state.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Executes an action in the environment.

        Args:
            action (int): Action index

        Returns:
            Tuple[np.ndarray, float, bool, Dict]: Tuple of next state, reward, terminal
            flag, and debugging dictionary.
        """
        raise NotImplementedError
