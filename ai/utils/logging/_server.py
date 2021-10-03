from typing import Any
import zmq
from queue import Empty
from multiprocessing import Process, Event

from torch.utils.tensorboard.writer import SummaryWriter

import ai.utils.logging as logging


class Server(Process):
    """Process handling a summary writer."""

    def __init__(self, *fields: logging.field.Base, name: str, port: int):
        """
        Args:
            fields (logging.field.Base): Logging fields.
            name (str): Name of the logger.
            port (int): Port on which the server will listen for logging values.
        """
        super().__init__(daemon=True)
        self._fields = {field.name: field for field in fields}
        self._writer: SummaryWriter = None
        self._port = port
        self._name = name

    def _log(self, field: str, value: Any):
        self._fields[field].log(self._writer, value)

    def run(self):
        self._writer = SummaryWriter(comment=self._name)
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.subscribe("")
        socket.bind(f"tcp://*:{self._port}")

        while True:
            if socket.poll(timeout=1.0, flags=zmq.POLLIN) != zmq.POLLIN:
                continue
            field, value = socket.recv()
            self._log(field, value)
