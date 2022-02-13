import abc
import torch


class Base(abc.ABC):
    """Base class from building distributions from environment state and model output."""

    @abc.abstractmethod
    def build(
        self, model_output: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.distributions.Distribution:
        """Builds a distribution.

        Args:
            state (torch.Tensor): Environment state, with zero or more batch dimensions.
            model_output (torch.Tensor): Output of model, with zero or more batch dimensions.

        Returns:
            torch.distributions.Distribution: Distribution.
        """
        pass
