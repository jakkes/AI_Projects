from typing import Tuple
import torch


class Config:
    """Trainer configuration."""

    def __init__(self) -> None:
        self.discount: float = 0.95
        """Discount factor."""

        self.batch_size: int = 32
        """Update batch size."""

        self.workers: int = 1
        """Number of asynchronous workers to run."""

        self.n_step: int = 1
        """Reward n step."""

        self.train_time: int = 600
        """Training time, in seconds."""
