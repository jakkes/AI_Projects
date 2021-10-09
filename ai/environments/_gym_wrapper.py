from typing import Tuple, Dict, Union

import numpy as np
import gym
from gym import spaces

import ai.environments as environments


class GymWrapper(environments.Base):
    """Environment wrapper for openAI gym environments."""

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

    def __init__(self, env: Union[gym.Env, str]):
        """
        Args:
            env (Union[gym.Env, str]): Environment instance to wrap, or the gym
                environment identifier..
        """
        super().__init__()
        self._env = gym.make(env) if type(env) is str else env
        self._action_space = GymWrapper.ActionSpace(self._env.action_space)

    @property
    def action_space(self) -> environments.action_spaces.Discrete:
        return self._action_space

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        return self._env.step(int(action))

    def reset(self) -> np.ndarray:
        return self._env.reset()

    def close(self):
        self._env.close()

    def render(self):
        self._env.render()
