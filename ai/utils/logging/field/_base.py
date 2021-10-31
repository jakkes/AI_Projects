import abc
from typing import Any
import torch.utils.tensorboard.writer


class Base(abc.ABC):
    """Base logging field."""
    
    def __init__(self, name: str):
        """
        Args:
            name (str): Name of the logging field.
        """
        super().__init__()
        self._name = name
        self._writer = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def writer(self) -> torch.utils.tensorboard.writer.SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, val):
        self._writer = val

    @abc.abstractmethod
    def log(self, value: Any):
        """Logs the item to the given writer."""
        pass
