from abc import abstractmethod

import numpy as np

import ai.environments.action_spaces as action_spaces


class Discrete(action_spaces.Base):
    """Discrete action space.

    Discrete action spaces identify actions using the an integer and have a fixed size.
    Moreover, all action are not necessarily legal in every state. Legal actions are
    given by the action mask, a boolean vector whose elements at legal action indices
    are `True` and illegal action indices `False`."""

    @property
    @abstractmethod
    def size(self) -> int:
        """The size of the discrete action space."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_mask(self) -> np.ndarray:
        """The boolean action mask of the current environmental state. Legal actions
        are marked with `True` and illegal actions with `False`."""
        raise NotImplementedError

    def sample(self) -> int:
        return np.random.choice(np.arange(self._size)[self.action_mask])

    def contains(self, action: int) -> bool:
        if not isinstance(action, (int, np.integer)):
            return False
        return self.action_mask[action]
