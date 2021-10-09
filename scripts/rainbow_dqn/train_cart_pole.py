import tap
import torch
from torch import optim, cuda

import ai.environments as environments
import ai.rl.dqn.rainbow as rainbow


class ArgumentParser(tap.Tap):
    episodes: int = 1000
    """Number of episodes to run the training for."""

    batch_size: int = 32
    """Batch size used by the agent."""

    target_update_steps: int = 10
    """Number of update steps between target network updates."""

    discount_factor: float = 0.99
    """Discount factor."""

    replay_capacity: int = 2**16
    """Replay capacity."""

    noise_std: float = 1.0
    """Initial noise STD in the NoisyNet."""


def main(args: ArgumentParser):
    env_factory = environments.GymWrapper.get_factory("CartPole-v0")

    device = torch.device("cpu") if cuda.is_available() else torch.device("cpu")

    agent_config = rainbow.AgentConfig()
    agent_config.action_space_size = 2
    agent_config.state_shape = (4,)
    agent_config.batch_size = args.batch_size
    agent_config.target_update_steps = args.target_update_steps
    agent_config.discount_factor = args.discount_factor
    agent_config.replay_capacity = args.replay_capacity
    agent_config.network_device = device

    agent_config.use_prioritized_experience_replay = False
    agent_config.use_distributional = True
    agent_config.use_double = True
    agent_config.n_atoms = 21
    agent_config.v_min = 0
    agent_config.v_max = 100

    network = rainbow.networks.CartPole(
        agent_config.use_distributional, agent_config.n_atoms, args.noise_std
    ).to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-4, )
    agent = rainbow.Agent(agent_config, network, optimizer)

    trainer_config = rainbow.trainers.basic.Config()
    trainer_config.episodes = args.episodes
    trainer_config.max_environment_steps = -1
    trainer_config.minimum_buffer_size = 200
    trainer_config.n_step = 3

    trainer = rainbow.trainers.basic.Trainer(agent, trainer_config, env_factory)
    trainer.start()


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
