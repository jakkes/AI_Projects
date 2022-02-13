from dataclasses import dataclass

import torch
from torch import nn, optim

from ai import environments


@dataclass
class BasicConfig:
    pass


class Basic:
    """Simple, synchronous, PPO trainer."""

    def __init__(
        self,
        config: BasicConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        env: environments.Base,
    ):
        """Creates a basic PPO trainer.

        Args:
            config (BasicConfig): Configuration.
            model (torch.nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer to 
            env (environments.Base): [description]
        """
        self._config = config
        self._model = model
        self._optimizer = optimizer
        self._env = env

    def run(self, duration: float):
        pass
