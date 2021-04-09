import attr


@attr.s(auto_attribs=True)
class Config:
    """Trainer config."""

    episodes: int = 100
    """Number of episodes to run."""

    max_environment_steps: int = -1
    """Maximum number of steps before an episode is terminated. If less than zero, this
    limit is not enforced."""

    n_step: int = 3
    """N-step rewards."""

    minimum_buffer_size: int = 100
    """Minimum buffer size before training steps are executed."""

    epsilon: float = 0.1
    """A random action is chosen, as opposed to the greedy action, with this
    probability during training."""
