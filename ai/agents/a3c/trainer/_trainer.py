from typing import Any, Mapping
from time import perf_counter, sleep

from torch import nn, optim


import ai.environments as environments
import ai.agents.a3c.trainer as trainer
from ._worker import Worker


class Trainer:
    """A3C trainer. Spawns multiple processes that each get a copy of the network."""

    def __init__(
        self,
        config: trainer.Config,
        environment: environments.Factory,
        network: nn.Module,
        optimizer_class: optim.Optimizer,
        optimizer_params: Mapping[str, Any],
    ):
        """
        Args:
            config (trainer.Config): Trainer configuration.
            environment (environments.Factory): Environment factory.
            network (nn.Module): Network with two outputs, policy logits and state
                value.
            optimizer_class (optim.Optimizer): Optimizer class.
            optimizer_params (Mapping[str, Any]): Keyword arguments sent to the
                optimizer class at initialization.
        """
        self._config = config
        network.share_memory()
        self._workers = [
            Worker(config, environment, network, optimizer_class, optimizer_params)
        ]

    def start(self):
        """Starts the training and blocks until it has finished."""
        for worker in self._workers:
            worker.start()

        start_time = perf_counter()
        while perf_counter() - start_time < self._config.train_time:
            sleep(5.0)
        
        for worker in self._workers:
            worker.terminate()
        for worker in self._workers:
            worker.join()
