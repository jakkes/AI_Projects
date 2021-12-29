from typing import List, Tuple

import torch

from ._base import Base


class RoundRobin(Base):
    """Combines multiple strategies, alternating between them using round robin.
    """
    def __init__(self, *strategies: Tuple[Base, ...]):
        self._strategies: List[Base] = list(strategies)
        self._i = 0

    def update(self, reward_sequences: torch.Tensor, sequence: bool=True):
        for strategy in self._strategies:
            strategy.update(reward_sequences, sequence=sequence)

    def reward_to_go(self) -> float:
        self._i = (self._i + 1) % len(self._strategies)
        return self._strategies[self._i].reward_to_go()
