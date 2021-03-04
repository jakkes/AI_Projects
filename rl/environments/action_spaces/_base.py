from abc import ABC, abstractmethod
from typing import Any, TypeVar

import rl

T = TypeVar("T")


class Base(ABC):

    def as_type(self, t: T) -> T:
        """Casts the action space to the specific type.

        Args:
            t (T): Type to which the space should be cast

        Raises:
            RuntimeError: If the object is not castable to the requested type.

        Returns:
            T: The same object.
        """
        if not isinstance(self, t):
            raise RuntimeError(f"Failed casting {self} to {t}")
        return self

    def as_discrete(self) -> "rl.environments.action_spaces.Discrete":
        """Casts this object to a discrete action space. This operation is equivalent
        to `as_type(DiscreteActionSpace)`."""
        return self.as_type(rl.environments.action_spaces.Discrete)

    @abstractmethod
    def sample(self) -> Any:
        """Samples an action from the action space.

        Returns:
            Any: An action.
        """
        raise NotImplementedError

    @abstractmethod
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
