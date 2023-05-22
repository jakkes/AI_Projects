import tap

import torch
from torch import nn, optim

from ai.utils import Factory
from ai import environments
from ai.rl.dqn import rainbow


class Args(tap.Tap):
    pass


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._seq = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4 * 101)
        )

    def forward(self, x):
        return torch.softmax(self._seq(x).view(-1, 4, 51), dim=-1)


def main(args: Args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    agent_config = rainbow.AgentConfig()
    agent_config.action_space_size = 4
    agent_config.state_shape = (8, )
    agent_config.batch_size = 128
    agent_config.discount_factor = 0.99
    agent_config.gradient_norm = 20
    agent_config.n_atoms = 51
    agent_config.network_device = device
    agent_config.replay_capacity = 1000000
    agent_config.replay_device = device
    agent_config.target_update_steps = 100
    agent_config.use_distributional = True
    agent_config.use_double = True
    agent_config.use_prioritized_experience_replay = False
    agent_config.v_min = -200
    agent_config.v_max = 200

    agent = rainbow.Agent(agent_config, Factory(Net), Factory(optim.Adam, lr=1e-4))

    trainer_config = rainbow.trainers.seed.Config()
    trainer_config.actor_processes = 10
    trainer_config.actor_threads = 32
    trainer_config.broadcast_period = 2.0
    trainer_config.epsilon = 0.01
    trainer_config.inference_batchsize = 128
    trainer_config.inference_delay = 0.5
    trainer_config.inference_device = device
    trainer_config.inference_servers = 1
    trainer_config.max_environment_steps = -1
    trainer_config.minimum_buffer_size = 50000
    trainer_config.n_step = 3
    trainer_config.max_train_frequency = -1
    
    trainer = rainbow.trainers.seed.Trainer(agent, trainer_config, environments.GymWrapper.get_factory("LunarLander-v2"))

    trainer.start(3600000)

if __name__ == "__main__":
    main(Args(underscores_to_dashes=True).parse_args())
