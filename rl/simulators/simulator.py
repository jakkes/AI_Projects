from abc import abstractclassmethod, ABC
from typing import Dict, List, Tuple

import numpy as np


State = Tuple[np.ndarray, ...]
States = Tuple[np.ndarray, ...]


class Simulator(ABC):
    """Simulator base class.

    A simulator, as opposed to an environment, executes actions based on a
    given state, rather than the interally tracked state. Thus, simulator
    classes are rarely (if ever) initialized.

    States are given by a tuple of `np.ndarray`s. Different implementations may use
    different number of arrays for defining their state. However, even if only one array
    is used, it is packed inside a tuple.

    Actions are, for now, only discrete and given by their action index.
    """

    @abstractclassmethod
    def action_mask_bulk(cls, states: States) -> np.ndarray:
        """Computes the legal actions of the given states.

        Args:
            states (States): States in batch format.

        Returns:
            np.ndarray: Boolean mask of legal actions.
        """
        raise NotImplementedError

    @classmethod
    def action_mask(cls, state: State) -> np.ndarray:
        """Computes the legal actions of the given state.

        Args:
            state (State): State

        Returns:
            np.ndarray: Boolean mask of legal actions
        """
        return cls.action_mask_bulk(tuple(np.expand_dims(x, 0) for x in state))[0]

    @classmethod
    def step(
        cls, state: State, action: int
    ) -> Tuple[State, np.ndarray, float, bool, Dict]:
        """Executes one step in the environment.

        Args:
            state (State): State
            action (int): Action index

        Returns:
            Tuple[State, np.ndarray, float, bool, Dict]: Tuple of next state,
            next action mask, reward, terminal flag, and debugging dictionary.
        """
        next_states, next_action_masks, rewards, terminals, infos = cls.step_bulk(
            np.expand_dims(state, 0), np.array([action])
        )
        return next_states[0], next_action_masks[0], rewards[0], terminals[0], infos[0]

    @abstractclassmethod
    def step_bulk(
        cls, states: States, actions: np.ndarray
    ) -> Tuple[States, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Executes a bulk of actions in multiple states.

        Args:
            states (States): States, in batch format.
            actions (np.ndarray): Integer vector of action indices.

        Returns:
            Tuple[States, np.ndarray, np.ndarray, np.ndarray, List[Dict]]: Tuple of
            next states, next action masks, rewards, terminal flags, and debugging
            dictionaries.
        """
        raise NotImplementedError

    @abstractclassmethod
    def reset_bulk(cls, n: int) -> Tuple[States, np.ndarray]:
        """Provides multiple new environment states.

        Args:
            n (int): Number of states to generate.

        Returns:
            Tuple[States, np.ndarray]: Tuple of states and boolean action masks.
        """
        raise NotImplementedError

    @classmethod
    def reset(cls) -> Tuple[State, np.ndarray]:
        """Provides a single new environment state.

        Returns:
            Tuple[State, np.ndarray]: Tuple of state and boolean action mask.
        """
        states, action_masks = cls.reset_bulk(1)
        return tuple(substates[0] for substates in states), action_masks[0]
