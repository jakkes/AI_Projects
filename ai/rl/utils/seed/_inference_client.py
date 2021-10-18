import io
import warnings

import zmq
import torch


class InferenceClient:
    """Client for running remote model inferences."""

    __slots__ = "_socket"

    def __init__(self, router_address: str):
        """
        Args:
            router_address (str): Address to `InferenceServer` instance, e.g.
                'tcp://127.0.0.1:33333`.
        """
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.connect(router_address)

    def evaluate_model(self, data: torch.Tensor, attempts: int = 5) -> torch.Tensor:
        """Runs a remote inference.

        Args:
            data (torch.Tensor): State.
            attempts (int, optional): Number of attempts made before an exception is
                raised. Defaults to 5.

        Returns:
            torch.Tensor: Inference result.
        """
        bytedata = io.BytesIO()
        torch.save(data, bytedata)
        self._socket.send(bytedata.getvalue())
        if self._socket.poll(timeout=5000, flags=zmq.POLLIN) != zmq.POLLIN:
            if attempts > 1:
                warnings.warn("Model evaluation failed, trying again...")
                return self.evaluate_model(data, attempts=attempts - 1)
            else:
                raise RuntimeError("Model evaluation failed.")
        recvbytes = io.BytesIO(self._socket.recv())
        return torch.load(recvbytes)
