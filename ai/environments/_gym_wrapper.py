from typing import Tuple, Dict

import numpy as np
import gym
from gym import spaces

import ai.environments as environments


class GymWrapper(environments.Base):
    """Environment wrapper for openAI gym environments."""

    class Factory(environments.Factory):
        def __init__(self, env_id: str):
            self.env_id = env_id

        def __call__(self) -> "GymWrapper":
            return GymWrapper(gym.make(self.env_id))

    class ActionSpace(environments.action_spaces.Discrete):
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

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Environment instance to wrap.
        """
        self._env = env
        self._action_space = GymWrapper.ActionSpace(env.action_space)

    @property
    def action_space(self) -> environments.action_spaces.Discrete:
        return self._action_space

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        action = int(action)
        return self._env.step(action)

    def reset(self) -> np.ndarray:
        return self._env.reset()

    def close(self):
        self._env.close()

    def render(self):
        self._env.render()

    @classmethod
    def get_factory(cls, env_id: str) -> "GymWrapper.Factory":
        """Creates an environment factory that spawns the specific `env_id`. For
        possible values, see the openAI gym documentation."""
        return GymWrapper.Factory(env_id)
