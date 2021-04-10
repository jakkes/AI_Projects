"""Action spaces for environments."""

from ._base import Base
from ._discrete import Discrete
from ._discrete_gym_wrapper import DiscreteGymWrapper


__all__ = ["Base", "Discrete", "DiscreteGymWrapper"]
