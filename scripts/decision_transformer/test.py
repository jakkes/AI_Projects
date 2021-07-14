import itertools

import torch
from torch import nn


from ai import environments
from ai.rl import decision_transformer as df


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.randn(x.shape[0], 4, dtype=torch.float16, device="cuda")


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 4)

    def forward(self, x):
        return self.linear(x.unsqueeze(1).half())


class RewardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 4)

    def forward(self, x):
        return self.linear(x.unsqueeze(1))


class PositionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 4)

    def forward(self, x):
        return self.linear(x.unsqueeze(1).half())


class ActionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def main():
    config = df.TrainerConfig((4,), 200, 1, 100, 10, 600, 5, True, True, 1)
    transformer = Transformer().half().cuda()
    action_decoder = ActionDecoder().half().cuda()
    state_encoder = StateEncoder().half().cuda()
    action_encoder = ActionEncoder().half().cuda()
    reward_encoder = RewardEncoder().half().cuda()
    postion_encoder = PositionEncoder().half().cuda()
    empty_embeddings = [
        torch.randn(4).half().cuda().requires_grad_() for _ in range(4)
    ]
    optimizer = torch.optim.Adam(
        itertools.chain(
            transformer.parameters(),
            state_encoder.parameters(),
            action_encoder.parameters(),
            reward_encoder.parameters(),
            postion_encoder.parameters(),
            empty_embeddings,
            action_decoder.parameters(),
        ),
        lr=1e-4,
    )
    trainer = df.Trainer(
        state_encoder,
        action_encoder,
        reward_encoder,
        postion_encoder,
        transformer,
        action_decoder,
        *empty_embeddings,
        environments.GymWrapper.Factory("CartPole-v0"),
        df.exploration_strategies.MaxObserved(0.0),
        config,
        optimizer
    )
    trainer.train()


if __name__ == "__main__":
    main()
