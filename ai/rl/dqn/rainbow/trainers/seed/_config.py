from dataclasses import dataclass

@dataclass
class Config:
    """Trainer config."""

    max_environment_steps: int = -1
    """Maximum number of steps before an episode is terminated. If less than zero, this
    limit is not enforced."""

    n_step: int = 3
    """N-step rewards."""

    epsilon: float = 0.1
    """Probability with which a random action is chosen during training."""

    actor_processes: int = 1
    """Number of actor processes spawned. Each actor process may spawn multiple actor
    threads."""

    actor_threads: int = 4
    """Number of actor threads spawned per actor process."""

    inference_servers: int = 1
    """Number of processes serving inference requests."""

    broadcast_period: float = 2.5
    """Period (seconds) between model parameter broadcasts."""

    inference_batchsize: int = 4
    """Maximum batch size of inference requests."""

    inference_delay: float = 0.1
    """Maximum delay of inference requests."""
