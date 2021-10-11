import io
import zmq
import torch


class Client:

    def __init__(self, router_address: str):
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.connect(router_address)

    def evaluate_model(self, data: torch.Tensor) -> torch.Tensor:
        bytedata = io.BytesIO()
        torch.save(data, bytedata)
        self._socket.send(bytedata.getvalue())
        if self._socket.poll(timeout=10000, flags=zmq.POLLIN) != zmq.POLLIN:
            raise RuntimeError("No reply received on inference request.")
        recvbytes = io.BytesIO(self._socket.recv())
        return torch.load(recvbytes)
