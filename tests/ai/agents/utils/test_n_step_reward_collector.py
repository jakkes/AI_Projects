import torch

import ai.agents as agents


def test_n_step_reward_collector():
    shapes = ((1,), (2,), (3,))
    dtypes = (torch.float32, torch.float32, torch.float32)
    collector = agents.utils.NStepRewardCollector(5, 0.75, shapes, dtypes)

    out = collector.step(
        1.0,
        False,
        tuple(torch.rand(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes))
    )
    assert out is None
