import numpy as np
from gym import spaces

import ai.environments.action_spaces as action_spaces


class DiscreteGymWrapper(action_spaces.Discrete):
    """Discrete action space that wraps a discrete openAI Gym action space."""

    def __init__(self, space: spaces.Discrete):
        """
        Args:
            space (spaces.Discrete): Space to wrap.
        """
        self._size = space.n

    @property
    def size(self) -> int:
        return self._size

    @property
    def action_mask(self) -> np.ndarray:
        return np.ones((self.size, ), dtype=np.bool_)
