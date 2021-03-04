import numpy as np

from ._discrete import Discrete


class TicTacToe(Discrete):
    """Action space for the TicTacToe simulator."""

    @classmethod
    def size(cls) -> int:
        return 9

    @classmethod
    def action_mask_bulk(cls, states: np.ndarray) -> np.ndarray:
        return states[:, :-1] == 0
