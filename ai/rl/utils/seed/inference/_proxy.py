import threading
from typing import Tuple

import zmq


def proxy_runner(router: zmq.Socket, dealer: zmq.Socket):
    zmq.proxy(router, dealer)


class Proxy:
    def __init__(self, router_port: int = None, dealer_port: int = None):
        self._router_port = router_port
        self._dealer_port = dealer_port
        self._thread: threading.Thread = None

    def start(self) -> Tuple[int, int]:
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
