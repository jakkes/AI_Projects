from typing import Any
from torch.utils.tensorboard.writer import SummaryWriter
from torch.multiprocessing import Queue
import ai.utils.logging as logging


class LearnerLogger(Server):
    """Logging server for the `LearnerWorker`."""
    def __init__(self, data_queue: Queue):
        """
        Args:
            data_queue (Queue): Queue from which logging items are gathered.
        """
        super().__init__("Learner", data_queue)
        self.step = 0

    def log(self, writer: SummaryWriter, data):
        self.step += 1
        writer.add_scalar("Training/Loss", data, global_step=self.step)


class SelfPlayLogger(Server):
    """Logging server for the `SelfPlayWorker`."""
    def __init__(self, data_queue: Queue):
        """
        Args:
            data_queue (Queue): Queue from which logging items are gathered.
        """
        super().__init__("SelfPlay", data_queue)
        self.step = 0

    def log(self, writer: SummaryWriter, data: Any):
        self.step += 1
        reward, start_value, kl_div, first_action = data
        writer.add_scalar("Episode/Reward", reward, global_step=self.step)
        writer.add_scalar(
            "Episode/Start value", start_value, global_step=self.step
        )
        writer.add_scalar(
            "Episode/Start KL Div", kl_div, global_step=self.step
        )
        writer.add_scalar(
            "Episode/First action", first_action, global_step=self.step
        )
