from typing import Any, Dict, Mapping, Tuple, NamedTuple
import abc

import torch
from torchaddons import distributions

import ai.environments as environments


Observation = NamedTuple(
    "Observation",
    [
        ("state", torch.Tensor),
        ("action_constraints", distributions.constraints.Base),
        ("reward", float),
        ("terminal", bool),
        ("debug_info", Mapping[str, Any])
    ]
)


class Base(abc.ABC):
    """Environment base class.

    An environment is a stateful environment on which actions may be executed. It has
    an internal state that is modified by the action and (potentially only partially)
    observable."""

    @abc.abstractmethod
    def reset(self) -> torch.Tensor:
        """Resets the environment to a new initial state.

        Returns:
            torch.Tensor: Initial state.
        """
        pass

    @abc.abstractmethod
    def step(self, action: torch.Tensor) -> Observation:
        """Executes an action in the environment.

        Args:
            action (torch.Tensor): Action.

        Returns:
            Observation: State transition
        """
        pass

    @abc.abstractmethod
    def close(self):
        """Disposes resources used by the environment."""
        pass

    @abc.abstractmethod
    def render(self):
        """Renders the environment in its current state."""
        raise NotImplementedError

    @classmethod
    def get_factory(cls, *args, **kwargs) -> "environments.Factory":
        """Creates and returns a factory object that spawns environments when called.

        Args and kwargs are passed along to the class constructor. However, if other
        behavior is required, this method may be overridden."""
        return environments.Factory(cls, *args, **kwargs)
