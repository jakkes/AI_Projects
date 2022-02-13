import pytest
import torch

from ai.rl.ppo import distribution_builder


def get_device(cuda: bool) -> torch.device:
    return torch.device("cuda", index=0) if cuda else torch.device("cpu")

def run_single(cuda: bool):
    device = get_device(cuda)

    builder = distribution_builder.Categorical(10)

    data = torch.randn(10, device=device)
    mask = torch.ones(10, dtype=torch.bool, device=device)
    mask[torch.randperm(10)[:3]] = False

    distribution = builder.build(data, mask)

    samples: torch.Tensor = distribution.sample((100,))
    
    assert samples.device == device
    assert mask[samples].all()


def test_single_cpu():
    run_single(False)

def test_single_cuda():
    if not torch.cuda.is_available():
        pytest.skip()

    run_single(True)


def run_batch(cuda: bool):
    device = get_device(bool)

    builder = distribution_builder.Categorical(10)

    data = torch.randn(13, 10, device=device)
    mask = torch.ones(13, 10, dtype=torch.bool, device=device)
    for i in range(13):
        mask[i, torch.randperm(10)[:3]] = False

    distribution = builder.build(data, mask)

    samples: torch.Tensor = distribution.sample((100,))
    
    assert samples.device == device
    for i in range(100):
        assert mask[torch.arange(13, device=device), samples[i]].all()

def test_batch_cpu():
    run_batch(False)

def test_batch_cuda():
    if not torch.cuda.is_available():
        pytest.skip()

    run_batch(True)
