import threading
from typing import Tuple

import zmq


def proxy_runner(router: zmq.Socket, dealer: zmq.Socket):
    zmq.proxy(router, dealer)


class InferenceProxy:
    """Acts as an intermediary between `InferenceClient`s and `InferenceServer`s."""

    def __init__(self, router_port: int = None, dealer_port: int = None):
        """
        Args:
            router_port (int, optional): Port to which clients connect. If `None`, then 
                a random port is selected on start. Defaults to None.
            dealer_port (int, optional): Port to which servers connect. If `None`, then
                a random port is selected on start. Defaults to None.
        """
        self._router_port = router_port
        self._dealer_port = dealer_port
        self._thread: threading.Thread = None

    def start(self) -> Tuple[int, int]:
        """Starts the proxy in a separate thread. The proxy will run until this object
        is garbage collected.

        Returns:
            Tuple[int, int]: Tuple of two ports. `InferenceClient`s connect to the first
                element, `InferenceServer`s connect to the second server.
        """
        router = zmq.Context.instance().socket(zmq.ROUTER)
        if self._router_port is None:
            self._router_port = router.bind_to_random_port("tcp://*")
        else:
            router.bind(f"tcp://*:{self._router_port}")

        dealer = zmq.Context.instance().socket(zmq.DEALER)
        if self._dealer_port is None:
            self._dealer_port = dealer.bind_to_random_port("tcp://*")
        else:
            dealer.bind(f"tcp://*:{self._dealer_port}")

        self._thread = threading.Thread(target=proxy_runner, args=(router, dealer), daemon=True)
        self._thread.start()

        return self._router_port, self._dealer_port
