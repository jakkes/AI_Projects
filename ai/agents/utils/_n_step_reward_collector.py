from typing import Tuple, Sequence

import torch


class NStepRewardCollector:
    """Utility object for collecting n-step rewards."""

    def __init__(self, n_step: int, discount_factor: float, shapes: Sequence[Tuple[int, ...]], dtypes: Sequence[torch.dtype]):
        """
        Args:
            n_step (int): N-step
            discount_factor (float): Discount factor
            shapes (Sequence[Tuple[int, ...]]): Shapes of data to store.
            dtypes (Sequence[torch.dtype]): Data types of data.
        """
        raise NotImplementedError
