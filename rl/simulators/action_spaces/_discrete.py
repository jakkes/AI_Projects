from abc import abstractclassmethod

import numpy as np

from ._base import Base


class Discrete(Base):
    """Discrete action space.

    Discrete action spaces identify actions using the an integer and have a fixed size.
    Moreover, all action are not necessarily legal in every state. Legal actions are
    given by the action mask, a boolean vector whose elements at legal action indices
    are `True` and illegal action indices `False`."""

    @abstractclassmethod
    def size(cls) -> int:
        """The size of the discrete action space."""
        raise NotImplementedError

    @classmethod
    def action_mask(cls, state: np.ndarray) -> np.ndarray:
        """Computes the action mask, indicating legal actions in the given state.

        Args:
            state (np.ndarray): State.

        Returns:
            np.ndarray: Action mask
        """
        return cls.action_mask_bulk(np.expand_dims(state, 0))[0]

    @abstractclassmethod
    def action_mask_bulk(cls, states: np.ndarray) -> np.ndarray:
        """Computes the action masks for the given states.

        Args:
            states (np.ndarray): States, where the first dimension is the batch
            dimension.

        Returns:
            np.ndarray: Action masks.
        """
        raise NotImplementedError

    def sample(cls, state: np.ndarray) -> int:
        return np.random.choice(np.arange(cls.size())[cls.action_mask(state)])

    def contains(cls, state: np.ndarray, action: int) -> bool:
        if not isinstance(action, (int, np.integer)):
            return False
        return cls.action_mask(state)[action]
