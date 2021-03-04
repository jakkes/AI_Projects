from abc import abstractmethod

import numpy as np

from ._base import Base


class Discrete(Base):

    def __init__(self, size: int) -> None:
        super().__init__()
        self._size = size

    @property
    def size(self):
        """The size of the discrete action space."""
        return self._size

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
