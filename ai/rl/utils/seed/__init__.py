"""Boilerplate for implementing distributed RL systems based on the SEED
architecture.

This module provides four components,
1. `Broadcaster`
2. `InferenceProxy`
3. `InferenceClient`
4. `InferenceServer`

These are covered below.

### Broadcaster
The `Broadcaster` class periodically broadcasts the latest model parameters. Example
```python
# Launches a Broadcaster instance to publish latest parameters of model every second.
model = torch.nn.Linear(5, 2)
broadcaster = Broadcaster(model, 1.0)
port = broadcaster.start()
```

### InferenceProxy
The `InferenceProxy` class launches a zeroMQ router-dealer proxy, allowing inference
servers and clients to communicate. Multiple servers and clients may connect to the
same proxy.

### InferenceServer
The `InferenceServer` class serves inference requests made from inference clients, 
routed through an inference proxy.

### InferenceClient
The `InferenceClient` class allows for sending inference requests to inference servers.

### Example
```python
import torch
from ai.rl.utils.seed import InferenceClient, InferenceProxy, InferenceServer, Broadcaster


if __name__ == "__main__":
    # Create simple model.
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 2),
        torch.nn.Softmax(dim=1)
    )

    # Create broadcaster, broadcasting the latest model parameters every 2.5 seconds.
    broadcaster = Broadcaster(model, 2.5)
    broadcasting_port = broadcaster.start()

    # Create inference proxy
    proxy = InferenceProxy()
    client_port, server_port = proxy.start()

    # Create some (three) inference servers
    servers = [
        InferenceServer(
            model,
            (5, ),
            torch.float32,
            f"tcp://127.0.0.1:{server_port}",
            f"tcp://127.0.0.1:{broadcasting_port}",
            2,
            0.1
        )
        for _ in range(3)
    ]
    for server in servers:
        server.start()

    # Create a client.
    client = InferenceClient(f"tcp://127.0.0.1:{client_port}")

    # Run some inferences
    for _ in range(10):
        y = client.evaluate_model(torch.randn(5))
        print(y)
```
"""


from ._broadcaster import Broadcaster
from ._inference_client import InferenceClient
from ._inference_server import InferenceServer
from ._inference_proxy import InferenceProxy
from ._data_collector import DataCollector
from ._data_publisher import DataPublisher


__all__ = [
    "Broadcaster",
    "InferenceClient",
    "InferenceServer",
    "InferenceProxy",
    "DataCollector",
    "DataPublisher",
]
