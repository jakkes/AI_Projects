import io
import threading

import torch
import zmq

from ai.utils import Metronome


def runner(model: torch.nn.Module, period: float, socket: zmq.Socket):
    metronome = Metronome(period)
    while True:
        metronome.wait()
        bytedata = io.BytesIO()
        torch.save(model.state_dict(), bytedata)
        socket.send(bytedata.getbuffer())


class Broadcaster:
    """An instance of this class will broadcast network parameters periodically for all
    inference servers to update their local models."""

    def __init__(self, model: torch.nn.Module, period: float, port: int = None):
        """
        Args:
            model (torch.nn.Module): Model whose state will periodically be broadcasted.
            period (float): Time (seconds) between consecutive broadcasts.
            port (int, optional): Port to which inference servers will connect to. If
                `None`, then a random port is selected. Defaults to None.
        """
        self._model = model
        self._period = period
        self._port = port
        self._thread: threading.Thread = None

    def start(self) -> int:
        """Starts the periodic broadcasting.

        Returns:
            int: Port on which the broadcasting is performed.
        """
        socket = zmq.Context.instance().socket(zmq.PUB)
        if self._port is None:
            self._port = socket.bind_to_random_port("tcp://*")
        else:
            socket.bind(f"tcp://*:{self._port}")
        self._thread = threading.Thread(
            target=runner, args=(self._model, self._period, socket), daemon=True
        )
        self._thread.start()
        return self._port
