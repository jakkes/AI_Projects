import torch.utils.tensorboard.writer
from ai.utils.logging.field import Base


class Scalar(Base):
    """Field logging scalar values."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__step = 0

    def log(self, writer: torch.utils.tensorboard.writer.SummaryWriter, value: float):
        writer.add_scalar(self.name, value, global_step=self.__step)
        self.__step += 1
