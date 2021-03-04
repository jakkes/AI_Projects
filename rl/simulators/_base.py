from abc import abstractclassmethod, ABC
from typing import Dict, List, Tuple

import numpy as np
from . import action_spaces


class Base(ABC):
    """Simulator base class.

    A simulator, as opposed to an environment, executes actions based on a
    given state, rather than the interally tracked state. Thus, simulator
    classes are rarely (if ever) initialized. States are given as `np.ndarray`s."""

    @classmethod
    def step(
        cls, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Executes one step in the environment.

        Args:
            state (np.ndarray): State
            action (int): Action index

        Returns:
            Tuple[np.ndarray, float, bool, Dict]: Tuple of next state, reward, terminal
            flag, and debugging dictionary.
        """
        next_states, rewards, terminals, infos = cls.step_bulk(
            np.expand_dims(state, 0), np.array([action])
        )
        return next_states[0], rewards[0], terminals[0], infos[0]

    @abstractclassmethod
    def action_space(cls) -> action_spaces.Base:
        """The action space class used by this simulator."""
        raise NotImplementedError

    @abstractclassmethod
    def step_bulk(
        cls, states: np.ndarray, actions: np.ndarray
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

    @abstractclassmethod
    def reset_bulk(cls, n: int) -> np.ndarray:
        """Provides multiple new environment states.

        Args:
            n (int): Number of states to generate.

        Returns:
            np.ndarray: Initial states, stacked in the first dimension.
        """
        raise NotImplementedError

    @classmethod
    def reset(cls) -> np.ndarray:
        """Provides a single new environment state.

        Returns:
            np.ndarray: Initial state
        """
        return cls.reset_bulk(1)[0]
