from typing import Type

import ai.environments as environments


class Factory():
    """Factories are callable objects that spawn environment instances."""

    __pdoc__ = {
        "Factory.__call__": True
    }

    def __init__(self, cls: Type[environments.Base], *args, **kwargs):
        """
        Args:
            cls (Type[environments.Base]): Environment class.
            `*args, **kwargs`: arguments and key-word arguments passed to the
                environment `__init__` method.
        """
        super().__init__()
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self) -> environments.Base:
        """Spawns and returns an environment instance.

        Returns:
            environments.Base: Instance of the environment.
        """
        return self._cls(*self._args, **self._kwargs)
