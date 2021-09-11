import queue
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
        self._logging_queue: queue.Queue = None

    def set_logging_queue(self, q: queue.Queue):
        """Sets the logging queue, into which logitems are passed by the environment.
        When environments are spawned through this factory object, their logging queue
        will be set to this value.

        Args:
            q (queue.Queue): Data queue. If `None`, logging is disabled.
        """
        self._logging_queue = q

    def __call__(self) -> environments.Base:
        """Spawns and returns an environment instance.

        Returns:
            environments.Base: Instance of the environment.
        """
        env = self._cls(*self._args, **self._kwargs)
        if self._logging_queue is not None:
            env.set_logging_queue(self._logging_queue)
        return env
