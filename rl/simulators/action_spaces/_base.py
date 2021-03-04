from abc import ABC, abstractclassmethod
from typing import Any, TypeVar, Tuple

import numpy as np
import rl

T = TypeVar("T")


class Base(ABC):
    """Base action space."""

    @classmethod
    def as_type(cls, t: T) -> T:
        """Casts the action space to the specific type.

        Args:
            t (T): Type to which the space should be cast

        Raises:
            RuntimeError: If the class is not castable to the requested type.

        Returns:
            T: Casted version of the class.
        """
        if not isinstance(cls, t):
            raise RuntimeError(f"Failed casting {cls} to {t}")
        return cls

    def as_discrete(cls) -> "rl.simulators.action_spaces.Discrete":
        """Casts this object to a discrete action space. This operation is equivalent
        to `as_type(DiscreteActionSpace)`."""
        return cls.as_type(rl.simulators.action_spaces.Discrete)

    @abstractclassmethod
    def sample(self, state: np.ndarray) -> Any:
        """Samples an action from the action space.

        Args:
            state (np.ndarray): State

        Returns:
            Any: An action.
        """
        raise NotImplementedError

    @abstractclassmethod
    def contains(cls, state: np.ndarray, action: Any) -> bool:
        """Determines if the given action is in the action space or not.

        Args:
            state (np.ndarray): State.
            action (Any): Action.

        Returns:
            bool: True if the given action is legal, otherwise False.
        """
        raise NotImplementedError

    def __contains__(cls, state_action: Tuple[np.ndarray, Any]) -> bool:
        return cls.contains(*state_action)
