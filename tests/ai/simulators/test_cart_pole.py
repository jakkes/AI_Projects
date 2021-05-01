import ai.simulators as simulators


def test_cart_pole():
    sim = simulators.CartPole()

    state = sim.reset()
    terminal = False
    while not terminal:
        state, _, terminal, _ = sim.step(sim.action_space.sample(state))
