import ai.environments as env
from ... import Agent
from ._config import Config


class Trainer:
    """Trainer class."""

    def __init__(self, agent: Agent, config: Config, environment: env.Base):
        self._agent = agent
        self._config = config
