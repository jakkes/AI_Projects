import torch
from ._base import Base


class MaxObserved(Base):
    """Exploration strategy returning maximum observed reward-to-go."""
    
    def __init__(self, min_value: float) -> None:
        """
        Args:
            min_value (float): Minimum reward-to-go to request. Also acts as start
                value.
        """
        super().__init__()
        self._value = min_value

    def update(self, reward_sequences: torch.Tensor):
        self._value = max(self._value, reward_sequences.sum(-1).max())

    def reward_to_go(self) -> float:
        return self._value
