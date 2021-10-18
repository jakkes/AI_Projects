import tap
import torch

import ai.environments as environments
from ai.rl.dqn.rainbow import AgentConfig, Agent, trainers, networks


class Args(tap.Tap):
    duration: float = 600
    """Train duration (seconds)."""

    batch_size: int = 32
    """Batch size used by the agent."""

    target_update_steps: int = 100
    """Number of update steps between target network updates."""

    discount_factor: float = 0.99
    """Discount factor."""

    replay_capacity: int = 10000
    """Replay capacity."""

    noise_std: float = 1.0
    """Initial noise STD in the NoisyNet."""


def main(args: Args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    agent_config = AgentConfig()
    agent_config.action_space_size = 2
    agent_config.state_shape = (4,)
    agent_config.batch_size = args.batch_size
    agent_config.target_update_steps = args.target_update_steps
    agent_config.discount_factor = args.discount_factor
    agent_config.replay_capacity = args.replay_capacity
    agent_config.network_device = device
    agent_config.replay_device = device
    agent_config.use_prioritized_experience_replay = False
    agent_config.use_distributional = True
    agent_config.use_double = True
    agent_config.n_atoms = 21
    agent_config.v_min = 0
    agent_config.v_max = 100

    network = networks.CartPole(
        agent_config.use_distributional, agent_config.n_atoms, args.noise_std
    ).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=1e-4,
    )
    agent = Agent(agent_config, network, optimizer)

    trainer_config = trainers.seed.Config()
    trainer_config.actor_processes = 4
    trainer_config.actor_threads = 1
    trainer_config.inference_batchsize = 4
    trainer_config.inference_delay = 0.1
    trainer_config.inference_device = device
    trainer_config.inference_servers = 1
    trainer_config.minimum_buffer_size = 1000
    trainer_config.n_step = 3
    trainer_config.epsilon = 0.1
    trainer_config.broadcast_period = 2.5

    trainer = trainers.seed.Trainer(
        agent, trainer_config, environments.GymWrapper.get_factory("CartPole-v0")
    )
    trainer.start(args.duration)


if __name__ == "__main__":
    args = Args(underscores_to_dashes=True).parse_args()
    main(args)
