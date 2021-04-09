import ai.environments as environments


def test_cart_pole():
    factory = environments.GymWrapper.get_factory("CartPole-v0")
    env = factory()

    state, terminal = env.reset(), False
    while not terminal:
        _, _, terminal, _ = env.step(env.action_space.sample())
