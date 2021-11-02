"""Decision Transformer, as proposed by Chen et al. https://arxiv.org/abs/2106.01345"""


from ._transformer import TransformerEncoder, Attention, MultiHeadAttention
from ._agent import Agent
from . import exploration_strategies
from ._trainer import Trainer
from ._trainer_config import TrainerConfig


__all__ = [
    "exploration_strategies",
    "Agent",
    "Trainer",
    "TrainerConfig",
    "TransformerEncoder",
    "Attention",
    "MultiHeadAttention"
]
