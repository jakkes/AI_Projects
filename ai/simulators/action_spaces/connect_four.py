import numpy as np

import ai.simulators.action_spaces as action_spaces


class ConnectFour(action_spaces.Discrete):
    """Action space of the ConnectFour simulator."""

    @property
    def size(cls) -> int:
        return 7

    def action_mask_bulk(self, states: np.ndarray) -> np.ndarray:
        return (states[:, :-1].reshape((-1, 6, 7)) != 0).sum(1) < 6
