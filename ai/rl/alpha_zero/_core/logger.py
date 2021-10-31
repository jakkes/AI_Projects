from typing import Any
from torch.utils.tensorboard.writer import SummaryWriter
from torch.multiprocessing import Queue
import ai.utils.logging as logging


class Logger(logging.Server):
    """Logging server for the `LearnerWorker`."""
    def __init__(self):
        """ """
        super().__init__(
            logging.field.Scalar("Training/Loss"),
            logging.field.Scalar("Episode/Reward"),
            logging.field.Scalar("Episode/Start value"),
            logging.field.Scalar("Episode/Start KL Div"),
            logging.field.Scalar("Episode/First action"),
            name="AlphaZero"
        )
