import abc
import torch


class Base(abc.ABC):
    """An exploration strategy defines the reward-to-go to be used at episode start.
    
    The exploration strategy is updated by observing reward sequences."""

    @abc.abstractmethod
    def update(self, reward_sequences: torch.Tensor, sequence: bool=True):
        """Updates the exploration strategy based on the observed reward sequences.

        Args:
            reward_sequences (torch.Tensor): Sequence or sequences of rewards. Last
                dimension contains the actual sequences.
            sequence (bool, optional): If True, a sequence of rewards are given from
                which the sum should be computed. If False, the summed rewards are
                expected. Defaults to True.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reward_to_go(self) -> float:
        """Returns a reward-to-go to be used in a specific episode.

        Returns:
            float: Reward-to-go, according to the exploration strategy.
        """
        raise NotImplementedError
