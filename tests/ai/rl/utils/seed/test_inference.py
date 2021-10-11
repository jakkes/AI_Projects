import time
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn

from ai.rl.utils.seed import inference, Broadcaster


def test_inference():
    model = nn.Sequential(
        nn.Linear(2, 10), nn.ReLU(inplace=True), nn.Linear(10, 1), nn.Sigmoid()
    )
    broadcaster = Broadcaster(model, 1.0)
    broadcast_port = broadcaster.start()

    proxy = inference.Proxy()
    router_port, dealer_port = proxy.start()

    inference_server = inference.Server(
        model,
        (2, ),
        torch.float32,
        f"tcp://127.0.0.1:{dealer_port}",
        f"tcp://127.0.0.1:{broadcast_port}",
        2,
        1.0
    )
    inference_server.start()

    time.sleep(10.0)

    inference_client1 = inference.Client(f"tcp://127.0.0.1:{router_port}")
    inference_client2 = inference.Client(f"tcp://127.0.0.1:{router_port}")
    inference_client3 = inference.Client(f"tcp://127.0.0.1:{router_port}")

    x1 = torch.randn(2)
    x2 = torch.randn(2)
    x3 = torch.randn(2)

    y1 = model(x1.unsqueeze(0))[0]
    y2 = model(x2.unsqueeze(0))[0]
    y3 = model(x3.unsqueeze(0))[0]
    
    f1 = lambda: inference_client1.evaluate_model(x1)
    f2 = lambda: inference_client2.evaluate_model(x2)
    f3 = lambda: inference_client3.evaluate_model(x3)

    with ThreadPoolExecutor(3) as pool:
        z1 = pool.submit(f1)
        z2 = pool.submit(f2)
        z3 = pool.submit(f3)

        z1 = z1.result(timeout=60.0)
        z2 = z2.result(timeout=60.0)
        z3 = z3.result(timeout=60.0)

    assert torch.all(z1 == y1)
    assert torch.all(z2 == y2)
    assert torch.all(z3 == y3)



if __name__ == "__main__":
    test_inference()
