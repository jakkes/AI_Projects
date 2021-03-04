from abc import abstractmethod, ABC, abstractclassmethod
from typing import Dict, List, Tuple

import numpy as np
import ai
from . import action_spaces


class Base(ABC):
    """Simulator base class.

    A simulator, as opposed to an environment, executes actions based on a
    given state, rather than a interally tracked state."""

    def step(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Executes one step in the environment.

        Args:
            state (np.ndarray): State
            action (int): Action index

        Returns:
            Tuple[np.ndarray, float, bool, Dict]: Tuple of next state, reward, terminal
            flag, and debugging dictionary.
        """
        next_states, rewards, terminals, infos = self.step_bulk(
            np.expand_dims(state, 0), np.array([action])
        )
        return next_states[0], rewards[0], terminals[0], infos[0]

    @property
    @abstractmethod
    def action_space(self) -> action_spaces.Base:
        """The action space class used by this simulator."""
        raise NotImplementedError

    @abstractmethod
    def step_bulk(
        self, states: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Executes a bulk of actions in multiple states.

        Args:
            states (np.ndarray): States, in batch format.
            actions (np.ndarray): Integer vector of action indices.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]: Tuple of
            next states, rewards, terminal flags, and debugging dictionaries.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_bulk(self, n: int) -> np.ndarray:
        """Provides multiple new environment states.

        Args:
            n (int): Number of states to generate.

        Returns:
            np.ndarray: Initial states, stacked in the first dimension.
        """
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        """Provides a single new environment state.

        Returns:
            np.ndarray: Initial state
        """
        return self.reset_bulk(1)[0]

    @abstractclassmethod
    def get_factory(cls, *args, **kwargs) -> "ai.simulators.Factory":
        """Creates and returns a factory object that spawns simulators when called."""
        raise NotImplementedError
