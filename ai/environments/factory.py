from typing import Generic, TypeVar


import ai.environments as environments


class Factory():
    """Factories are callable objects that spawn environment instances."""

    def __init__(self, cls: environments.Base, *args, **kwargs):
        """
        Args:
            cls (T): Environment class.
            *args, **kwargs: arguments and key-word arguments passed to the environment
                __init__ method.
        """
        super().__init__()
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self) -> environments.Base:
        return self._cls(*self._args, **self._kwargs)
