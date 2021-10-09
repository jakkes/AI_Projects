from typing import Type, Optional

import ai.environments as environments
import ai.utils.logging as logging


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
        self._logging_client: logging.Client = None

    def set_logging_client(self, client: Optional[logging.Client]):
        """Sets the logging client, into which logitems are passed by the environment.
        When environments are spawned through this factory object, their logging clients
        will be set to this value.

        Args:
            client (Optional[logging.Client]): Logging client. If `None`, then 
                logging is disabled.
        """
        self._logging_client = client

    def __call__(self) -> environments.Base:
        """Spawns and returns an environment instance.

        Returns:
            environments.Base: Instance of the environment.
        """
        env = self._cls(*self._args, **self._kwargs)
        env.logging_client = self._logging_client
        return env
