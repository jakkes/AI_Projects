from typing import Dict, Tuple, TypeVar
from abc import ABC, abstractmethod

import numpy as np

import ai
import ai.environments as environments


T = TypeVar("T")


class Base(ABC):
    """Environment base class.

    An environment is a stateful environment upon which action may be executed. It has
    an internal state that is modified by the action and (potentially only partially)
    observable from the outside."""

    @property
    @abstractmethod
    def action_space(self) -> environments.action_spaces.Base:
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

    @classmethod
    def get_factory(cls: T, *args, **kwargs) -> "ai.environments.Factory[T]":
        """Creates and returns a factory object that spawns simulators when called.

        Args and kwargs are passed along to the class constructor. However, if other
        behavior is required, feel free to override this method and return a factory
        class of your choice."""
        return ai.environments.Factory[T](cls, *args, **kwargs)
