import queue
import multiprocessing
from typing import Any, Optional

from torch.utils.tensorboard.writer import SummaryWriter

import ai

import torch

from ._base import Base


class MaxObserved(Base):
    """Exploration strategy returning maximum observed reward-to-go."""
    
    def __init__(self, min_value: float, logging_queue: Optional[queue.Queue] = None) -> None:
        """
        Args:
            min_value (float): Minimum reward-to-go to request. Also acts as start
                value.
            logging_queue (optional, queue.Queue): If given, logging information is
                sent to this queue.
        """
        super().__init__()
        self._value = min_value
        self._logging_queue = logging_queue

    def update(self, reward_sequences: torch.Tensor):
        self._value = max(self._value, reward_sequences.sum(-1).max())
        if self._logging_queue is not None:
            self._logging_queue.put(self._value)

    def reward_to_go(self) -> float:
        return self._value

    class LoggingServer(ai.utils.logging.SummaryWriterServer):
        def __init__(self, data_queue: multiprocessing.Queue):
            super().__init__("exploration_strategy", data_queue)

        def log(self, summary_writer: SummaryWriter, data: float):
            summary_writer.add_scalar("Exploration Strategy/Value", data)
