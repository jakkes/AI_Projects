import abc
import torch.utils.tensorboard.writer


class Base(abc.ABC):
    """Base logging item."""
    
    def __init__(self) -> None:
        """ """
        super().__init__()

    @abc.abstractmethod
    def log(self, writer: torch.utils.tensorboard.writer.SummaryWriter):
        """Logs the item to the given writer."""
        raise NotImplementedError
