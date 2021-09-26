import itertools
import multiprocessing as mp

import torch
from torch import nn


from ai import environments
from ai.rl import decision_transformer as df


EMBED_SPACE = 8


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self._lstm = nn.LSTM(
            input_size=EMBED_SPACE,
            hidden_size=EMBED_SPACE,
            num_layers=4,
            batch_first=True,
        )

    def forward(self, x):
        h, c = self._lstm(x)
        return h[:, -1]


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(inplace=True), nn.Linear(32, EMBED_SPACE)
        )

    def forward(self, x):
        return self.seq(x)


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, EMBED_SPACE)
        )

    def forward(self, x):
        return self.seq(x.unsqueeze(-1))


class RewardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, EMBED_SPACE)
        )

    def forward(self, x):
        return self.seq(x.unsqueeze(-1))


class PositionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, EMBED_SPACE)
        )

    def forward(self, x):
        return self.seq(x.unsqueeze(-1))


class ActionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(EMBED_SPACE, 32), nn.ReLU(inplace=True), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.seq(x)


def main():
    config = df.TrainerConfig(
        state_shape=(4,),
        action_size=2,
        batch_size=1,
        max_episode_steps=200,
        number_of_actors=6,
        replay_capacity=10000,
        min_replay_size=5000,
        training_time=3600,
        inference_sequence_length=5,
        enable_float16=True,
        enable_cuda=True,
        inference_batchsize=5
    )
    transformer = Transformer().half().cuda()
    action_decoder = ActionDecoder().half().cuda()
    state_encoder = StateEncoder().half().cuda()
    action_encoder = ActionEncoder().half().cuda()
    reward_encoder = RewardEncoder().half().cuda()
    postion_encoder = PositionEncoder().half().cuda()
    empty_embeddings = [torch.zeros(EMBED_SPACE).half().cuda().requires_grad_() for _ in range(4)]
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
        eps=1e-6,
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
