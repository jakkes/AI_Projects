import threading
import torch.utils.tensorboard.writer

from ai.utils.logging.field import Base
from ai.utils import Metronome


def timer(self: "Frequency"):
    step = 0
    metronome = Metronome(self._log_period)
    while True:
        metronome.wait()
        self.writer.add_scalar(self.name, self._n / self._log_period, global_step=step)
        self._n = 0
        step += 1


class Frequency(Base):
    """Field logging frequency of occurances."""

    __slots__ = '_thread', '_log_period', '_n'

    def __init__(self, name: str, log_period: float) -> None:
        """
        Args:
            name (str): Name of logging field.
            log_period (float): Period (seconds) between logs.
        """
        super().__init__(name)
        self._thread = None
        self._log_period = log_period
        self._n = 0

    def log(self, occurances: int):
        self._n += occurances

        if self._thread is None:
            self._thread = threading.Thread(target=timer, args=(self, ), daemon=True)
            self._thread.start()
