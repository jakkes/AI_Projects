from typing import Callable
import torch

from . import distribution_builder


"""Agent class for PPO inferences."""


class Agent:
    def __init__(
        self,
        model: torch.nn.Module,
        distribution_builder: distribution_builder.Base,
    ) -> None:
        """Constructs a PPO inference object.

        Args:
            model (torch.nn.Module): Underlying model.
            distribution_builder (distribution_builder.Base): Function converting the
                state and model output to a torch distribution from which the action is
                sampled.
        """
        self._model = model
        self._distribution_builder = distribution_builder

    def act(self, state: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Samples an action according to the policy.

        Args:
            state (torch.Tensor): State(s).
            action_mask (torch.Tensor): Tensor containing action restrictions, passed
                along with model output to the distribution builder.

        Returns:
            torch.Tensor: Tensor containing the sampled action.
        """
        with torch.no_grad():
            return self._distribution_builder.build(
                self._model(state.unsqueeze(0)).squeeze_(0),
                action_mask
            ).sample()
