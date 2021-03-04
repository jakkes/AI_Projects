from time import perf_counter, sleep

from torch import nn, optim
from torch.multiprocessing import Queue

import ai.simulators as simulators
from . import (
    LearnerConfig,
    LearnerWorker,
    SelfPlayLogger,
    SelfPlayConfig,
    SelfPlayWorker,
    LearnerLogger,
)


def train(
    simulator: simulators.Factory,
    self_play_workers: int,
    learner_config: LearnerConfig,
    self_play_config: SelfPlayConfig,
    network: nn.Module,
    optimizer: optim.Optimizer,
    save_path: str = None,
    save_period: int = -1,
    train_time: int = -1
):
    """Starts training an AlphaZero model.

    Args:
        simulator (simulators.Factory): Simulator factory spawning simulators
        on which to train the model.
        self_play_workers (int): Number of self play workers to spawn.
        learner_config (LearnerConfig): Configuration for the learner worker.
        self_play_config (SelfPlayConfig): Configuration for the self play worker.
        network (nn.Module): Network.
        optimizer (optim.Optimizer): Optimizer.
        save_path (str, optional): Path to where to store training checkpoints. If None,
        no checkpoints are stored. Defaults to None.
        save_period (int, optional): Time (in seconds) between checkpoints. If less than
        zero, no saves are made. Defaults to -1.
        train_time (int, optional): Training time in seconds. If less than zero,
        training is run until the process is interupted. Defaults to -1.
    """
    network.share_memory()

    sample_queue = Queue(maxsize=2000)
    episode_logging_queue = Queue(maxsize=2000)
    learner_logging_queue = Queue(maxsize=2000)

    self_play_workers = [
        SelfPlayWorker(
            simulator,
            network,
            self_play_config,
            sample_queue,
            episode_logging_queue=episode_logging_queue,
        )
        for _ in range(self_play_workers)
    ]

    learner_worker = LearnerWorker(
        network,
        optimizer,
        learner_config,
        sample_queue,
        learner_logging_queue=learner_logging_queue,
        save_path=save_path,
        save_period=save_period
    )

    learner_logger = LearnerLogger(learner_logging_queue)
    self_play_logger = SelfPlayLogger(episode_logging_queue)

    learner_logger.start()
    self_play_logger.start()
    learner_worker.start()
    for worker in self_play_workers:
        worker.start()

    start = perf_counter()
    while train_time < 0 or perf_counter() - start < train_time:
        sleep(10)

    learner_logger.terminate()
    self_play_logger.terminate()
    learner_worker.terminate()
    for worker in self_play_workers:
        worker.terminate()

    learner_logger.join()
    self_play_logger.join()
    learner_worker.join()
    for worker in self_play_workers:
        worker.join()
