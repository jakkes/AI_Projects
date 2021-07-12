"""Decision Transformer, as proposed by Chen et al. https://arxiv.org/abs/2106.01345"""


from . import exploration_strategies
from ._trainer import Trainer
from ._trainer_config import TrainerConfig


__all__ = ["exploration_strategies", "Trainer", "TrainerConfig"]
