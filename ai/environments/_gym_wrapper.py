from typing import Tuple, Dict

import numpy as np
import gym

import ai.environments as environments


class Factory(environments.Factory):
    def __init__(self, env_id: str):
        self.env_id = env_id

    def __call__(self) -> "GymWrapper":
        return GymWrapper(gym.make(self.env_id))


class GymWrapper(environments.Base):
    """Environment wrapper for openAI gym environments."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Environment instance to wrap.
        """
        self._env = env
        self._action_space = environments.action_spaces.DiscreteGymWrapper(
            env.action_space
        )

    @property
    def action_space(self) -> environments.action_spaces.Base:
        return self._action_space

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        return self._env.step(action)

    def reset(self) -> np.ndarray:
        return self._env.reset()

    def close(self):
        self._env.close()

    @classmethod
    def get_factory(cls, env_id: str) -> Factory:
        """Creates an environment factory that spawns the specific `env_id`. For
        possible values, see the openAI gym documentation."""
        return Factory(env_id)
