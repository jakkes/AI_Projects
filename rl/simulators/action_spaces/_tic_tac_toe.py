import numpy as np

from ._discrete import Discrete


class TicTacToe(Discrete):

    def __init__(self) -> None:
        super().__init__()

    def size(cls) -> int:
        return 9

    def action_mask_bulk(cls, states: np.ndarray) -> np.ndarray:
        return states[:, :-1] == 0
