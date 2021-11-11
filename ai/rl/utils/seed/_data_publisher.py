import io
import zmq
import torch


class DataPublisher:
    def __init__(self, address: str):
        self._address = address
        self._pub = zmq.Context.instance().socket(zmq.PUB)
        self._pub.connect(self._address)

    def publish(self, *tensors: torch.Tensor):
        buffer = io.BytesIO()
        torch.save(tensors, buffer)
        self._pub.send(buffer.getbuffer(), copy=False)

    def __getstate__(self):
        return self._address

    def __setstate__(self, data):
        self._address = data
        self._pub = zmq.Context.instance().socket(zmq.PUB)
        self._pub.connect(self._address)

    def clone(self) -> "DataPublisher":
        return DataPublisher(self._address)
