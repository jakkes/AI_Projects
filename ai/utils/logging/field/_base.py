import abc
from typing import Any
import torch.utils.tensorboard.writer


class Base(abc.ABC):
    """Base logging field."""
    
    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): Name of the logging field.
        """
        super().__init__()
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

    @abc.abstractmethod
    def log(self, writer: torch.utils.tensorboard.writer.SummaryWriter, value: Any):
        """Logs the item to the given writer."""
        pass
