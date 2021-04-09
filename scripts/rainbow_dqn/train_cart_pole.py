import tap
from torch import nn, optim

import ai.environments as environments
import ai.agents.dqn.rainbow as rainbow


class ArgumentParser(tap.Tap):
    episodes: int = 100
    """Number of episodes to run the training for."""


def main(args: ArgumentParser):
    env_factory = environments.GymWrapper.get_factory("CartPole-v0")

    agent_config = rainbow.AgentConfig()
    agent_config.action_space_size = 2
    agent_config.state_shape = (4, )
    agent_config.batch_size = 32
    agent_config.target_update_steps = 10
    agent_config.discount_factor = 0.99
    agent_config.replay_capacity = 10000

    agent_config.use_prioritized_experience_replay = False
    agent_config.use_distributional = False
    agent_config.use_double = False

    network = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 2)
    )
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    agent = rainbow.Agent(agent_config, network, optimizer)

    trainer_config = rainbow.trainers.basic.Config()
    trainer_config.episodes = args.episodes
    trainer_config.max_environment_steps = -1
    trainer_config.minimum_buffer_size = 200
    trainer_config.n_step = 1

    trainer = rainbow.trainers.basic.Trainer(agent, trainer_config, env_factory)
    trainer.start()


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
