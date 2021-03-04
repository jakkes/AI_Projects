"""Utility methods dealing with random numbers in PyTorch."""

import torch


@torch.jit.script
def choice(probabilities: torch.Tensor) -> torch.Tensor:
    """Selects elements with probability proportional to the given weights.

    Args:
        probabilities (torch.Tensor): Tensor of shape `(..., N)`. Arguments are sampled
        over the last dimension according to the weight.

    Returns:
        torch.Tensor: Tensor of shape equal to the input shape, but the last dimension.
        Data type is `torch.long`.

    Example usage:
    ```python
    x = torch.tensor([[1.0, 1.0, 1.0],
                      [0.0, 1.0, 2.0]])
    y = choice(x)   # y[0] is any of (0, 1, 2) with equal probability. y[1] is any of
                    # (1, 2), with 2 occuring with twice the probability of 1.
    ```
    """
    cumsummed = (probabilities / probabilities.sum(-1, keepdim=True)).cumsum(-1)
    r = torch.rand(probabilities.shape[:-1]).unsqueeze_(-1)
    return (r > cumsummed).sum(-1)
