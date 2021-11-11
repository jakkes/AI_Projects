import torch

from ._base import Base


class Quantile(Base):
    """Reward-to-go strategy returning a certain quantile of last seen rewards."""

    def __init__(self, quantile: float, learning_rate: float=1.0, momentum: float=0.99) -> None:
        """
        Args:
            quantile (float): Quantile, float in (0, 1).
            learning_rate (float, optional): Learning rate used in the gradient descent
                algorithm. Defaults to 1.0.
            momentum (float, optional): Momentum used in the gradient descent algorithm.
                Defaults to 0.99.
            
        """
        super().__init__()
        self.__quantile = quantile
        self.__alpha = learning_rate
        self.__momentum_rate = momentum
        self.__momentum = 0.0
        self.__value = 0.0
        self.__step = 0

    def update(self, reward_sequences: torch.Tensor, sequence: bool=True):
        self.__step += 1
        
        rtg = reward_sequences.sum(-1) if sequence else reward_sequences
        self.__momentum += (1 - self.__momentum_rate) * (self.__quantile - (rtg < self.__value).float().mean() - self.__momentum)
        self.__value += self.__alpha * self.__momentum / (1 - self.__momentum_rate ** self.__step)

    def reward_to_go(self) -> float:
        return self.__value