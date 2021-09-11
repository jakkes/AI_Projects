import torch.utils.tensorboard.writer
from ai.utils.logging.items import Base


class Scalar(Base):
    """A logging item containing a scalar value."""

    def __init__(self, name: str, value: float) -> None:
        """A logging item containing a scalar value.
        
        Args:
            name (str): Logging field name
            value (float): Value
        """
        super().__init__()
        self._name = name
        self._value = value

    def log(self, writer: torch.utils.tensorboard.writer.SummaryWriter):
        writer.add_scalar(self._name, self._value)
