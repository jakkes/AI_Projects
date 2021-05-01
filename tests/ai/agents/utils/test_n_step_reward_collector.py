import torch

import ai.agents as agents


def test_simple():
    shapes = ((1,), (2,))
    dtypes = (torch.float32, torch.float32)
    collector = agents.utils.NStepRewardCollector(3, 0.75, shapes, dtypes)

    for _ in range(3):
        out = collector.step(
            1.0,
            False,
            tuple(
                torch.rand(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)
            ),
        )
        assert out is None

    out = collector.step(
        1.0,
        False,
        tuple(torch.rand(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)),
    )
    assert out is not None


def test_n_step_3():
    shapes = ((),)
    dtypes = (torch.long,)
    collector = agents.utils.NStepRewardCollector(3, 0.5, shapes, dtypes)

    all_states = torch.arange(20)

    for i in range(3):
        assert collector.step(i, False, (all_states[i],)) is None

    for i in range(3, 20):
        (states, ), rewards, terminals, (next_states, ) = collector.step(
            i, False, (all_states[i],)
        )
        assert states.shape == (1, )
        assert states[0] == i - 3
        assert next_states.shape == (1, )
        assert next_states[0] == i
        assert terminals.shape == (1, )
        assert terminals[0].item() is False
        assert rewards.shape == (1, )
        assert rewards[0] == (i - 3) + 0.5 * (i - 2) + 0.5**2 * (i - 1)


def test_terminal():
    shapes = ((),)
    dtypes = (torch.long,)
    collector = agents.utils.NStepRewardCollector(3, 0.5, shapes, dtypes)

    all_states = torch.arange(100)

    for i in range(1, 4):
        assert collector.step(i, False, (all_states[i],)) is None

    for i in range(4, 9):
        (states, ), rewards, terminals, (next_states, ) = collector.step(
            i, False, (all_states[i],)
        )
        assert states.shape == (1, )
        assert states[0] == i - 3
        assert next_states.shape == (1, )
        assert next_states[0] == i
        assert terminals.shape == (1, )
        assert terminals[0].item() is False
        assert rewards.shape == (1, )
        assert rewards[0] == (i - 3) + 0.5 * (i - 2) + 0.5**2 * (i - 1)

    (states, ), rewards, terminals, (next_states, ) = collector.step(
        9, True, (all_states[9],)
    )
    assert (states == torch.arange(6, 10)).all()
    assert next_states[0] == all_states[9]
    assert (rewards == torch.tensor([
        6 + 0.5 * 7 + 0.5**2 * 8,
        7 + 0.5 * 8 + 0.5**2 * 9,
        8 + 0.5 * 9,
        9
    ])).all()

    for i in range(10, 13):
        assert collector.step(i, False, (all_states[i],)) is None

    for i in range(13, 20):
        (states, ), rewards, terminals, (next_states, ) = collector.step(
            i, False, (all_states[i],)
        )
        assert states.shape == (1, )
        assert states[0] == i - 3
        assert next_states.shape == (1, )
        assert next_states[0] == i
        assert terminals.shape == (1, )
        assert terminals[0].item() is False
        assert rewards.shape == (1, )
        assert rewards[0] == (i - 3) + 0.5 * (i - 2) + 0.5**2 * (i - 1)


def test_n_step_1():
    shapes = ((),)
    dtypes = (torch.long,)
    collector = agents.utils.NStepRewardCollector(1, 0.5, shapes, dtypes)

    all_states = torch.arange(20)

    for i in range(1):
        assert collector.step(i, False, (all_states[i],)) is None

    for i in range(1, 20):
        (states, ), rewards, terminals, (next_states, ) = collector.step(
            i, False, (all_states[i],)
        )
        assert states.shape == (1, )
        assert states[0] == i - 1
        assert next_states.shape == (1, )
        assert next_states[0] == i
        assert terminals.shape == (1, )
        assert terminals[0].item() is False
        assert rewards.shape == (1, )
        assert rewards[0] == i - 1


def test_clear():
    shapes = ((),)
    dtypes = (torch.long,)
    collector = agents.utils.NStepRewardCollector(3, 0.5, shapes, dtypes)

    all_states = torch.arange(20)

    for i in range(3):
        assert collector.step(i, False, (all_states[i],)) is None

    for i in range(3, 10):
        (states, ), rewards, terminals, (next_states, ) = collector.step(
            i, False, (all_states[i],)
        )
        assert states.shape == (1, )
        assert states[0] == i - 3
        assert next_states.shape == (1, )
        assert next_states[0] == i
        assert terminals.shape == (1, )
        assert terminals[0].item() is False
        assert rewards.shape == (1, )
        assert rewards[0] == (i - 3) + 0.5 * (i - 2) + 0.5**2 * (i - 1)

    collector.clear()
    for i in range(10, 13):
        assert collector.step(i, False, (all_states[i],)) is None
    
    for i in range(13, 20):
        (states, ), rewards, terminals, (next_states, ) = collector.step(
            i, False, (all_states[i],)
        )
        assert states.shape == (1, )
        assert states[0] == i - 3
        assert next_states.shape == (1, )
        assert next_states[0] == i
        assert terminals.shape == (1, )
        assert terminals[0].item() is False
        assert rewards.shape == (1, )
        assert rewards[0] == (i - 3) + 0.5 * (i - 2) + 0.5**2 * (i - 1)
