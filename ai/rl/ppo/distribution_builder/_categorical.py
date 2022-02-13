import torch

from ._base import Base


class Categorical(Base):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self._dim = dim

    def build(self, model_output: torch.Tensor, action_mask: torch.Tensor) -> torch.distributions.Distribution:
        """Builds a categorical distribution.

        Args:
            model_output (torch.Tensor): Output of model, probability logits.
            action_mask (torch.Tensor): Boolean action mask.

        Returns:
            torch.distributions.Distribution: Policy
        """
        model_output[~action_mask] = -torch.inf
        return torch.distributions.Categorical(logits=model_output)
