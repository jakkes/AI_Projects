import dataclasses
from typing import Tuple


@dataclasses.dataclass
class TrainerConfig:
    state_shape: Tuple[int, ...]
    """Shape of environment states."""

    max_episode_steps: int
    """Maximum number of steps one episode may consist of. This parameter has an
    immediate effect on memory usage, as the replay buffer is scaled accordingly."""

    number_of_actors: int
    """Number of actors to run in data collection."""

    replay_capacity: int
    """Number of sequences in the replay buffer."""

    min_replay_size: int
    """Minimum size of the replay buffer before training starts."""

    training_time: int
    """Training time, in seconds."""

    inference_sequence_length: int
    """Length of sequences passed to models, known as K in the original paper."""
