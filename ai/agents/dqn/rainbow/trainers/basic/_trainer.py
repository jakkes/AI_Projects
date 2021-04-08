import ai.environments as env
from ... import Agent
from ._config import Config


class Trainer:
    """Trainer class."""

    def __init__(self, agent: Agent, config: Config, environment: env.Factory):
        """
        Args:
            agent (Agent): Agent.
            config (Config): Trainer configuration.
            environment (env.Factory): Environment factory.
        """
        self._agent = agent
        self._config = config
        self._env_factory = environment

    def start(self):
        """Starts training, according to the configuration."""
