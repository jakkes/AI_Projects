import time

import torch
import ai.rl.utils.seed as seed


def test_data_pub():
    collector = seed.DataCollector(10, ((5, ), ), (torch.float32, ))
    port = collector.start()
    time.sleep(1.0)

    publisher = seed.DataPublisher(f"tcp://127.0.0.1:{port}")
    time.sleep(1.0)

    x = torch.randn(5)
    y = torch.randn(5)

    publisher.publish(x)
    publisher.publish(y)
    time.sleep(1.0)

    assert collector.size == 2
    data, _, _ = collector.get_all()
    assert data[0][0].allclose(x)
    assert data[0][1].allclose(y)

