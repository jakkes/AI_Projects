from abc import ABC, abstractclassmethod
from typing import Any, TypeVar

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
            RuntimeError: If the object is not castable to the requested type.

        Returns:
            T: Casted version of the class.
        """
        if not isinstance(cls, t):
            raise RuntimeError(f"Failed casting {cls} to {t}")
        return cls

    def as_discrete(self) -> "rl.simulators.action_spaces.Discrete":
        """Casts this object to a discrete action space. This operation is equivalent
        to `as_type(DiscreteActionSpace)`."""
        return self.as_type(rl.simulators.action_spaces.Discrete)

    @abstractclassmethod
    def sample(self) -> Any:
        """Samples an action from the action space.

        Returns:
            Any: An action.
        """
        raise NotImplementedError

    @abstractclassmethod
    def contains(self, action: Any) -> bool:
        """Determines if the given action is in the action space or not.

        Args:
            action (Any): Action.

        Returns:
            bool: True if the given action is legal, otherwise False.
        """
        raise NotImplementedError

    def __contains__(self, action: Any) -> bool:
        return self.contains(action)
