import numpy as np

from ._discrete import Discrete


class TicTacToe(Discrete):
    """Action space for the TicTacToe simulator."""

    @property
    def size(cls) -> int:
        return 9

    def action_mask_bulk(self, states: np.ndarray) -> np.ndarray:
        return states[:, :-1] == 0
