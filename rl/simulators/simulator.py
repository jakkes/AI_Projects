from abc import abstractclassmethod, ABC
from typing import Dict, List, Tuple

import numpy as np


class Simulator(ABC):
    """Simulator base class.

    A simulator, as opposed to an environment, executes actions based on a
    given state, rather than the interally tracked state. Thus, simulator
    classes are rarely (if ever) initialized.

    States are given as `np.ndarray`s. Actions are, for now, only discrete and given by
    their action index. States are always returned with a corresponding action mask,
    indicating which actions are legal in the given state.
    """

    @classmethod
    def step(
        cls, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """Executes one step in the environment.

        Args:
            state (np.ndarray): State
            action (int): Action index

        Returns:
            Tuple[np.ndarray, np.ndarray, float, bool, Dict]: Tuple of next state,
            next action mask, reward, terminal flag, and debugging dictionary.
        """
        next_states, next_action_masks, rewards, terminals, infos = cls.step_bulk(
            np.expand_dims(state, 0), np.array([action])
        )
        return next_states[0], next_action_masks[0], rewards[0], terminals[0], infos[0]

    @abstractclassmethod
    def step_bulk(
        cls, states: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Executes a bulk of actions in multiple states.

        Args:
            states (np.ndarray): States, in batch format.
            actions (np.ndarray): Integer vector of action indices.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]: Tuple of
            next states, next action masks, rewards, terminal flags, and debugging
            dictionaries.
        """
        raise NotImplementedError

    @abstractclassmethod
    def reset_bulk(cls, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Provides multiple new environment states.

        Args:
            n (int): Number of states to generate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of states and boolean action masks.
        """
        raise NotImplementedError

    @classmethod
    def reset(cls) -> Tuple[np.ndarray, np.ndarray]:
        """Provides a single new environment state.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of state and boolean action mask.
        """
        states, action_masks = cls.reset_bulk(1)
        return states[0], action_masks[0]
