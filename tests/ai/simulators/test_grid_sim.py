import numpy as np

import ai.simulators as simulators


def test_reset():
    sim = simulators.Grid(3, (3, 2, 4))
    states = sim.reset_bulk(5)
    assert states.shape == (5, 2, 3)


def test_step():
    sim = simulators.Grid(2, (2, 2))
    state = np.array([[1, 0], [0, 1]])
    state, _, _, _ = sim.step(state, 2)
    assert np.array_equal(state, np.array([[1, 1], [0, 1]]))
    state, _, terminal, _ = sim.step(state, 1)
    assert np.array_equal(state, np.array([[0, 1], [0, 1]]))
    assert terminal


def test_grid():
    sim = simulators.Grid(2, (2, 2))
    state = sim.reset()
    terminal = False
    for _ in range(1000):
        state, _, terminal, _ = sim.step(state, sim.action_space.sample(state))
        if terminal:
            break
    assert terminal
