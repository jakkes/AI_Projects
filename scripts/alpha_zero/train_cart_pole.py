from torch import nn, optim

import ai.simulators as simulators
import ai.agents.alpha_zero as alpha_zero


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self._body = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True)
        )
        self._policy = nn.Linear(32, 2)
        self._value = nn.Linear(32, 1)

    def forward(self, states, action_masks):
        x = self._body(states)
        return self._policy(x), self._value(x)


def main():

    learner_config = alpha_zero.LearnerConfig()
    learner_config.batch_size = 16

    self_play_config = alpha_zero.SelfPlayConfig()
    self_play_config.zero_sum_game = False
    self_play_config.simulations = 10
    self_play_config.c = 50

    network = Network()
    optimizer = optim.Adam(network.parameters())

    alpha_zero.train(
        simulators.CartPole.get_factory(),
        2,
        learner_config,
        self_play_config,
        network,
        optimizer
    )


if __name__ == "__main__":
    main()
