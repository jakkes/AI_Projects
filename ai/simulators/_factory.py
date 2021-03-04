from abc import ABC, abstractmethod
import ai.simulators as simulators


class Factory(ABC):
    """Factories are callable objects that spawn simulator instances."""

    @abstractmethod
    def __call__(self) -> "simulators.Base":
        raise NotImplementedError
