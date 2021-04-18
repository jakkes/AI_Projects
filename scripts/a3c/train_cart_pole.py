from torch import nn, optim

import ai
import ai.agents.a3c as a3c


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Linear(32, 1)
        self.policy = nn.Linear(32, 2)

    def forward(self, x):
        x = self.body(x)
        return self.policy(x), self.value(x)


if __name__ == "__main__":
    config = a3c.trainer.Config()
    config.batch_size = 32
    config.discount = 0.95
    config.n_step = 3
    config.train_time = 30
    config.workers = 8

    trainer = a3c.trainer.Trainer(
        config,
        ai.environments.GymWrapper.get_factory("CartPole-v0"),
        Network(),
        optim.Adam,
        {"lr": 1e-4}
    )
    trainer.start()
