import abc
import numpy as np


class Heuristic(abc.ABC):
    """Base class for heuristic functions. Heuristics evaluate a given state and
    indicates who is (believed to be) in favor. The heuristic needs to be symmetric.
    Given a state with assigned value `X`, the exact opposite state must be assigned
    `-X`. This is due to the zero-sum assumption.
    """

    @abc.abstractmethod
    def __call__(self, state: np.ndarray) -> float:
        """Computes a heuristic value of the given state.

        Args:
            state (np.ndarray): State.

        Returns:
            float: Heuristic value, indicating which player is in favor.
        """
        raise NotImplementedError
