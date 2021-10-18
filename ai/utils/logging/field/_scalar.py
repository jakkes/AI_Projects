import torch.utils.tensorboard.writer
from ai.utils.logging.field import Base


class Scalar(Base):
    """Field logging scalar values."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._step = 0

    def log(self, value: float):
        self.writer.add_scalar(self.name, value, global_step=self._step)
        self._step += 1
