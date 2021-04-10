"""Action spaces for environments."""

from .base import Base
from .discrete import Discrete
from .discrete_gym_wrapper import DiscreteGymWrapper


__all__ = ["Base", "Discrete", "DiscreteGymWrapper"]
