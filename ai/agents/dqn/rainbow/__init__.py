"""Rainbow DQN."""


from ._agent import Agent
from ._agent_config import AgentConfig
from . import networks, trainers


__all__ = ["Agent", "AgentConfig", "networks", "trainers"]
