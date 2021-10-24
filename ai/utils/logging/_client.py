from typing import Any
import threading

import zmq


class Client:
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._socket = zmq.Context.instance().socket(zmq.PUB)
        self._socket.connect(f"tcp://{self._host}:{self._port}")
        self._lock = threading.Lock()

    def __getstate__(self):
        return self._host, self._port

    def __setstate__(self, data):
        self._host, self._port = data
        self._socket = zmq.Context.instance().socket(zmq.PUB)
        self._socket.connect(f"tcp://{self._host}:{self._port}")
        self._lock = threading.Lock()

    def log(self, field: str, value: Any):
        with self._lock:
            self._socket.send_pyobj((field, value))
