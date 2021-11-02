import dataclasses
from typing import Tuple

import torch


@dataclasses.dataclass
class TrainerConfig:
    state_shape: Tuple[int, ...]
    """Shape of environment states."""

    action_size: int
    """Size of action space, i.e. number of available actions in total."""

    batch_size: int = 64
    """Number of sequences to use during each update step."""

    max_episode_steps: int = -1
    """Maximum number of steps one episode may consist of. This parameter has an
    immediate effect on memory usage, as the replay buffer is scaled accordingly."""

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

    replay_capacity: int = 10000
    """Number of sequences in the replay buffer."""

    min_replay_size: int = 1000
    """Minimum size of the replay buffer before training starts."""

    replay_device: torch.device = torch.device("cpu")
    """Device on which the replay buffer should be located."""

    inference_sequence_length: int = 15
    """Length of sequences passed to models, known as K in the original paper."""

    enable_float16: bool = False
    """If True, training is run using `torch.float16` tensors, otherwise
    `torch.float32`"""

    network_device: torch.device = torch.device("cpu")
    """Device on which the network is to be located."""

    max_train_frequency: float = -1
    """Maximum number of training steps per second. Negative value results in no limit.
    This may be useful if data collection is slow."""

    max_environment_steps: int = -1
    """Maximum number of steps before an episode is terminated. If less than zero, this
    limit is not enforced."""
