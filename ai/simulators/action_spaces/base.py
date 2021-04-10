from abc import ABC, abstractmethod
from typing import Any, TypeVar, Tuple

import numpy as np
import ai.simulators.action_spaces as action_spaces

T = TypeVar("T")


class Base(ABC):
    """Base simulator action space."""

    def cast_to(self, t: T) -> T:
        """Casts the action space to the specific type.

        Args:
            t (T): Type to which the space should be cast

        Raises:
            RuntimeError: If the object is not castable to the requested type.

        Returns:
            T: Casted version of the object.
        """
        if not isinstance(self, t):
            raise RuntimeError(f"Failed casting {self} to {t}")
        return self

    @property
    def as_discrete(self) -> "action_spaces.Discrete":
        """This object cast to a discrete action space. Equivalent to calling
        `cast_to(Discrete)`."""
        return self.cast_to(action_spaces.Discrete)

    @abstractmethod
    def sample(self, state: np.ndarray) -> Any:
        """Samples an action from the action space.

        Args:
            state (np.ndarray): State

        Returns:
            Any: An action.
        """
        raise NotImplementedError

    @abstractmethod
    def contains(self, state: np.ndarray, action: Any) -> bool:
        """Determines if the given action is in the action space or not.

        Args:
            state (np.ndarray): State.
            action (Any): Action.

        Returns:
            bool: True if the given action is legal, otherwise False.
        """
        raise NotImplementedError

    def __contains__(self, state_action: Tuple[np.ndarray, Any]) -> bool:
        return self.contains(*state_action)
