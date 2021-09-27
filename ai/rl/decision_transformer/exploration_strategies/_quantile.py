import torch

from ._base import Base


class Quantile(Base):
    """Reward-to-go strategy returning a certain quantile of last seen rewards."""

    def __init__(self, quantile: float, update_rate: float=0.9) -> None:
        """
        Args:
            quantile (float): Quantile, float in (0, 1).
            update_rate (float, optional): Float in (0, 1). Controls the smoothing of
                the quantile regression. Higher value increases smoothing. Defaults to
                0.99.
        """
        super().__init__()
        self.__quantile = quantile
        self.__alpha = 1 - update_rate
        self.__value = 0.0

    def update(self, reward_sequences: torch.Tensor):
        rtg = reward_sequences.sum(-1)
        self.__value += self.__alpha * (self.__quantile - (rtg < self.__value).float().mean())

    def reward_to_go(self) -> float:
        return self.__value