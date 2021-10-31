from typing import Any, Mapping
from time import perf_counter, sleep

from torch import nn, optim


import ai.environments as environments
import ai.rl.a3c.trainer as trainer
from ._worker import Worker
from ._logger import Logger


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
        self._environment = environment
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params
        self._network = network
        network.share_memory()
        self._workers: List[Worker] = []
        self._logger = Logger()

    def start(self):
        """Starts the training and blocks until it has finished."""

        log_port = self._logger.start()

        self._workers = [
            Worker(
                self._config,
                self._environment,
                self._network,
                self._optimizer_class,
                self._optimizer_params,
                log_port,
            )
            for _ in range(self._config.workers)
        ]

        for worker in self._workers:
            worker.start()

        start_time = perf_counter()
        while perf_counter() - start_time < self._config.train_time:
            sleep(5.0)

        for worker in self._workers:
            worker.terminate()
        self._logger.terminate()
        for worker in self._workers:
            worker.join()
        self._logger.join()
