from argparse import ArgumentParser

from torch import optim

from ai.simulators import ConnectFour
from ai.rl.alpha_zero import networks, train, LearnerConfig, SelfPlayConfig


parser = ArgumentParser()


if __name__ == "__main__":
    args = parser.parse_args()

    net = networks.ConnectFourNetwork()
    train(
        ConnectFour.get_factory(),
        8,
        LearnerConfig(),
        SelfPlayConfig(),
        net,
        optim.Adam(net.parameters(), lr=1e-3),
        save_path="models/alpha_zero/connect_four",
        save_period=60,
        train_time=300,
    )
