from typing import Type
import ai.simulators as simulators


class Factory:
    """Factories are callable objects that spawn simulator instances."""

    __pdoc__ = {
        "Factory.__call__": True
    }

    def __init__(self, cls: Type[simulators.Base], *args, **kwargs):
        """
        Args:
            cls (Type[simulators.Base]): Simulator class.
            `*args, **kwargs`: arguments and key-word arguments passed to the simulator
                `__init__` method.
        """
        super().__init__()
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self) -> simulators.Base:
        """Spawns and returns a simulator instance.

        Returns:
            simulators.Base: Instance of the simulator.
        """
        return self._cls(*self._args, **self._kwargs)
