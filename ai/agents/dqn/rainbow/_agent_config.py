from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class AgentConfig:

    state_shape: Tuple[int, ...] = attr.Factory(tuple)
    """Shape of the state space."""

    action_space_size: int = 0
    """Number of actions in the action space."""

    replay_capacity: int = 10000
    """Capacity of the replay buffer."""

    batch_size: int = 32
    """Batch size used in learning steps."""

    target_update_steps: int = 20
    """Number of update steps to apply between each target network update."""

    discount_factor: float = 0.99
    """Discount factor."""

    use_double: bool = True
    """Whether or not to use Double DQN."""

    use_distributional: bool = True
    """Whether or not to use the distributional part of RainbowDQN."""

    n_atoms: int = 51
    """Number of support atoms to use in the distributional part of RainbowDQN."""

    v_min: float = -1
    """Minimum value of the distribution support."""

    v_max: float = 1
    """Maximum value of the distribution support."""

    use_prioritized_experience_replay: bool = True
    """Whether or not to use the prioritized experience replay part of RainbowDQN."""

    alpha: float = 0.6
    """Controls the distribution of the prioritized experience replay."""

    beta_start: float = 0.4
    """Start value of the beta parameter, used in prioritized experience replay."""

    beta_end: float = 1.0
    """End value of the beta parameter, used in prioritized experience replay."""

    beta_t_start: int = 0
    """Number of updates to apply before linearly annealing beta from `beta_start` to
    `beta_end`."""

    beta_t_end: int = 10000
    """Number of updates after which beta should have annealed to `beta_end`."""
