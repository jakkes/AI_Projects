import queue
from typing import Mapping

from torch.utils.tensorboard.writer import SummaryWriter

import ai.utils.logging as logging


class Logger(logging.SummaryWriterServer):
    def __init__(self, data_queue: queue.Queue):
        super().__init__("Trainer", data_queue)
        self.agent_step = 0
        self.episode = 0

    def log(self, summary_writer: SummaryWriter, data: Mapping):
        if "loss" in data and "max_error" in data:
            summary_writer.add_scalar("Agent/Loss", data["loss"], self.agent_step)
            summary_writer.add_scalar(
                "Agent/Max error", data["max_error"], self.agent_step
            )
            self.agent_step += 1
        if "reward" in data and "discounted_reward" in data:
            summary_writer.add_scalar("Env/Reward", data["reward"], self.episode)
            summary_writer.add_scalar(
                "Env/Discounted reward", data["discounted_reward"], self.episode
            )
            summary_writer.add_scalar(
                "Env/Steps", data["steps"], self.episode
            )
            summary_writer.add_scalar(
                "Agent/Start value", data["start_value"], self.episode
            )
            self.episode += 1
