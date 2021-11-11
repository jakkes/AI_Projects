from typing import Callable, Tuple, List
import io
import threading

import zmq
import torch

from ai.utils import logging
import ai.rl.utils.buffers as buffers


class DataCollector(buffers.Uniform):
    def __init__(
        self,
        capacity: int,
        shapes: Tuple[Tuple[int, ...]],
        dtypes: Tuple[torch.dtype],
        device: torch.device = torch.device("cpu"),
        log_client: logging.Client = None,
        port: int = None
    ):
        super().__init__(capacity, shapes, dtypes, device=device)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._port = port
        self._sub = zmq.Context.instance().socket(zmq.SUB)
        self._sub.subscribe("")
        self._log_client = log_client

    def _run(self):
        step = 0

        while True:
            if self._sub.poll(timeout=1000) != zmq.POLLIN:
                continue
            buffer = io.BytesIO(self._sub.recv())
            data = torch.load(buffer)
            self.add(data, 1.0, batch=False)

            step += 1
            if step == 100:
                step = 0
                if self._log_client is not None:
                    self._log_client.log("Data rate", 100)

    def start(self) -> int:
        if self._port is None:
            self._port = self._sub.bind_to_random_port("tcp://*")
        else:
            self._sub.bind(f"tcp://*:{self._port}")
        
        self._thread.start()
        return self._port
