import numpy as np

import ai.simulators.action_spaces as action_spaces


class TicTacToe(action_spaces.Discrete):
    """Action space for the TicTacToe simulator."""

    @property
    def size(cls) -> int:
        return 9

    def action_mask_bulk(self, states: np.ndarray) -> np.ndarray:
        return states[:, :-1] == 0
