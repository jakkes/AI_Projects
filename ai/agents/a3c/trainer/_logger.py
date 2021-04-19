import queue

from torch.utils.tensorboard.writer import SummaryWriter
import ai.utils.logging as logging


class Logger(logging.SummaryWriterServer):
    def __init__(self, data_queue: queue.Queue):
        super().__init__("a3c", data_queue)

    def log(self, summary_writer: SummaryWriter, data):
        if "r" in data:
            summary_writer.add_scalar("Episode/Reward", data["r"])
        if "l" in data:
            summary_writer.add_scalar("Agent/Loss", data["l"])
