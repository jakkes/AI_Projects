"""Basic sequential trainer. A training step is executed every _n_th environment
step."""


from ._trainer import Trainer
from ._config import Config


__all__ = ["Trainer", "Config"]
