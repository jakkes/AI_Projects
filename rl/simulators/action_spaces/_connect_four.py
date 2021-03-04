import numpy as np

from ._discrete import Discrete


class ConnectFour(Discrete):
    """Action space of the ConnectFour simulator."""

    @classmethod
    def size(cls) -> int:
        return 7

    @classmethod
    def action_mask_bulk(cls, states: np.ndarray) -> np.ndarray:
        return (states[:, :-1].reshape((-1, 6, 7)) != 0).sum(1) < 6
