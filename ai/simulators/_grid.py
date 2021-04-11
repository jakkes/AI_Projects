from typing import Optional, Sequence, Tuple
import ai.simulators as simulators


class ActionSpace(simulators.action_spaces.Discrete):
    def __init__(self, dim: int):
        super().__init__()
        self._dim = dim

    @property
    def size(self) -> int:
        return self._dim * 2


class Grid(simulators.Base):
    """Simple grid navigation environment. Agents can move in either direction in all
    dimensions and need to reach a goal state.

    States are given by two vectors, specifying the grid coordinates of the agent and
    goal respectively. A grid of dimension `N` therefore has a state shape of
    `(2, N)`"""

    def __init__(
        self, dim: int, boundaries: Sequence[Tuple[Optional[int], Optional[int]]]
    ):
        super().__init__(True)
