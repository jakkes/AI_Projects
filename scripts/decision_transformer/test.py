import torch
from torch import nn


from ai import environments
from ai.rl import decision_transformer as dt
from ai.utils import Factory


EMBED_SPACE = 8


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Linear(4, EMBED_SPACE)

    def forward(self, x):
        return self.seq(x)


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Linear(1, EMBED_SPACE)

    def forward(self, x):
        return self.seq(x.float().unsqueeze(-1))


class RewardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Linear(1, EMBED_SPACE)

    def forward(self, x):
        return self.seq(x)


class PositionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Linear(1, EMBED_SPACE)

    def forward(self, x):
        return self.seq(x)


class ActionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(EMBED_SPACE, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.seq(x).squeeze(-1)


def main():
    config = dt.TrainerConfig(
        state_shape=(4,),
        action_shape=(),
        discrete_action_space=True,
        max_environment_steps=200,
        min_replay_size=100
    )
    transformer = Factory(dt.TransformerEncoder, 8, 8, EMBED_SPACE, 8, 8, 1024)
    action_decoder = Factory(ActionDecoder)
    state_encoder = Factory(StateEncoder)
    action_encoder = Factory(ActionEncoder)
    reward_encoder = Factory(RewardEncoder)
    position_encoder = Factory(PositionEncoder)
    optimizer = Factory(torch.optim.Adam, lr=1e-4)

    agent = dt.Agent(
        state_encoder,
        action_encoder,
        reward_encoder,
        position_encoder,
        transformer,
        action_decoder
    )

    trainer = dt.Trainer(
        agent,
        optimizer,
        config,
        environments.GymWrapper.get_factory("CartPole-v0"),
        dt.exploration_strategies.Quantile(0.75)
    )
    trainer.train(3600)


if __name__ == "__main__":
    main()
