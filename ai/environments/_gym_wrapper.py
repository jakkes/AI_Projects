from typing import Tuple, Dict, Union

import torch
from torchaddons import distributions
import gym
from gym import spaces

import ai.environments as environments


class GymWrapper(environments.Base):
    """Environment wrapper for openAI gym environments."""

    def __init__(self, env: Union[gym.Env, str]):
        """
        Args:
            env (Union[gym.Env, str]): Environment instance to wrap, or the gym
                environment identifier..
        """
        super().__init__()
        self._env: gym.Env = gym.make(env) if type(env) is str else env
        self._constraint: distributions.constraints.Base = None

        if isinstance(self._env.action_space, spaces.Discrete):
            self._constraint = distributions.constraints.CategoricalMask(torch.ones(self._env.action_space.n, dtype=torch.bool))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        return self._env.step(int(action))

    def reset(self) -> np.ndarray:
        return self._env.reset()

    def close(self):
        self._env.close()

    def render(self):
        self._env.render()
