import io
import zmq
import torch


class InferenceClient:
    """Client for running remote model inferences."""

    def __init__(self, router_address: str):
        """
        Args:
            router_address (str): Address to `InferenceServer` instance, e.g.
                'tcp://127.0.0.1:33333`.
        """
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.connect(router_address)

    def evaluate_model(self, data: torch.Tensor) -> torch.Tensor:
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
            raise RuntimeError("No reply received on inference request.")
        recvbytes = io.BytesIO(self._socket.recv())
        return torch.load(recvbytes)
