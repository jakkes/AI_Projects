import warnings
import io

import zmq
import torch


class InferenceClient:
    """Client for running remote model inferences."""

    __slots__ = "_socket", "_router_address"

    def __init__(self, router_address: str):
        """
        Args:
            router_address (str): Address to `InferenceServer` instance, e.g.
                'tcp://127.0.0.1:33333`.
        """
        self._router_address = router_address
        self._socket: zmq.Socket = None
        self._create_socket()

    def _create_socket(self):
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.connect(self._router_address)

    def evaluate_model(self, data: torch.Tensor, max_attempts: int=5) -> torch.Tensor:
        """Runs a remote inference.

        Args:
            data (torch.Tensor): State.

        Returns:
            torch.Tensor: Inference result.
        """
        bytedata = io.BytesIO()
        torch.save(data, bytedata)
        self._socket.send(bytedata.getvalue())
        if self._socket.poll(timeout=10000, flags=zmq.POLLIN) != zmq.POLLIN:
            if max_attempts == 1:
                raise RuntimeError("No reply received on inference request.")
            else:
                warnings.warn("Inference request failed, trying again...")
                self._create_socket()
                return self.evaluate_model(data, max_attempts - 1)
        recvbytes = io.BytesIO(self._socket.recv())
        return torch.load(recvbytes)
