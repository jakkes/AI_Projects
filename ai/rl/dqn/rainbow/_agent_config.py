import torch
from typing import Tuple


class AgentConfig:
    """RainbowDQN agent configuration."""
    def __init__(self) -> None:
        """ """

        self.state_shape: Tuple[int, ...] = tuple()
        """Shape of the state space."""

        self.action_space_size: int = 0
        """Number of actions in the action space."""

        self.replay_capacity: int = 10000
        """Capacity of the replay buffer."""

        self.batch_size: int = 32
        """Batch size used in learning steps."""

        self.target_update_steps: int = 20
        """Number of update steps to apply between each target network update."""

        self.discount_factor: float = 0.99
        """Discount factor."""

        self.use_double: bool = True
        """Whether or not to use Double DQN."""

        self.use_distributional: bool = True
        """Whether or not to use the distributional part of RainbowDQN."""

        self.n_atoms: int = 51
        """Number of support atoms to use in the distributional part of RainbowDQN."""

        self.v_min: float = -1
        """Minimum value of the distribution support."""

        self.v_max: float = 1
        """Maximum value of the distribution support."""

        self.use_prioritized_experience_replay: bool = True
        """Whether or not to use the prioritized experience replay part of RainbowDQN."""

        self.alpha: float = 0.6
        """Controls the distribution of the prioritized experience replay."""

        self.beta_start: float = 0.4
        """Start value of the beta parameter, used in prioritized experience replay."""

        self.beta_end: float = 1.0
        """End value of the beta parameter, used in prioritized experience replay."""

        self.beta_t_start: int = 0
        """Number of updates to apply before linearly annealing beta from `beta_start` to
        `beta_end`."""

        self.beta_t_end: int = 10000
        """Number of updates after which beta should have annealed to `beta_end`."""

        self.gradient_norm: float = 20
        """Gradients are normed (L2) to this value, if larger. If this value is negative,
        no normalization is done."""

        self.network_device: torch.device = torch.device("cpu")
        """Device on which the network lives. Samples are moved to this device before
        ran through the network. Defaults to CPU."""
