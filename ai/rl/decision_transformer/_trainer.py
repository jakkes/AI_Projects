import signal
import threading
from typing import List, Tuple
from multiprocessing.connection import Connection

import torch
from torch import nn, optim, multiprocessing as mp

from ai import environments
from . import exploration_strategies
from ._trainer_config import TrainerConfig


class Trainer:
    """Trainer object, training a Decision Transformer model."""

    def __init__(
        self,
        state_encoder: nn.Module,
        action_encoder: nn.Module,
        reward_encoder: nn.Module,
        positional_encoder: nn.Module,
        transformer: nn.Module,
        state_empty_embedding: torch.Tensor,
        action_empty_embedding: torch.Tensor,
        reward_empty_embedding: torch.Tensor,
        position_empty_embedding: torch.Tensor,
        environment: environments.Factory,
        exploration_strategy: exploration_strategies.Base,
        config: TrainerConfig,
        optimizer: optim.Optimizer,
    ):
        self._state_encoder = state_encoder.share_memory()
        self._action_encoder = action_encoder.share_memory()
        self._reward_encoder = reward_encoder.share_memory()
        self._positional_encoder = positional_encoder.share_memory()
        self._transformer = transformer.share_memory()
        self._state_empty_embedding = state_empty_embedding.share_memory_()
        self._action_empty_embedding = action_empty_embedding.share_memory_()
        self._reward_empty_embedding = reward_empty_embedding.share_memory_()
        self._empty_positional_embedding = position_empty_embedding.share_memory_()
        self._optimizer = optimizer
        self._config = config
        self._exploration_strategy = exploration_strategy
        self._environment = environment

        self._stop_training = threading.Event()

    def stop(self):
        """Signals for training to stop."""
        self._stop_training.set()

    def train(self):
        """Starts training. Blocks until training is terminated."""
        pass


class Inference(mp.Process):
    def __init__(
        self,
        state_encoder: nn.Module,
        action_encoder: nn.Module,
        reward_encoder: nn.Module,
        positional_encoder: nn.Module,
        transformer: nn.Module,
        state_empty_embedding: torch.Tensor,
        action_empty_embedding: torch.Tensor,
        reward_empty_embedding: torch.Tensor,
        position_empty_embedding: torch.Tensor,
        environment: environments.Factory,
        exploration_strategy: exploration_strategies.Base,
        config: TrainerConfig,
    ):
        super().__init__()
        self._state_encoder = state_encoder.share_memory()
        self._action_encoder = action_encoder.share_memory()
        self._reward_encoder = reward_encoder.share_memory()
        self._positional_encoder = positional_encoder.share_memory()
        self._transformer = transformer.share_memory()
        self._state_empty_embedding = state_empty_embedding.share_memory_()
        self._action_empty_embedding = action_empty_embedding.share_memory_()
        self._reward_empty_embedding = reward_empty_embedding.share_memory_()
        self._empty_positional_embedding = position_empty_embedding.share_memory_()
        self._config = config
        self._exploration_strategy = exploration_strategy
        self._environment = environment

        self._inference_threads: List[threading.Thread] = []
        self._sigterm_detected: threading.Event = None

    def _sigterm(self, *args, **kwargs):
        self._sigterm_detected.set()

    def _inference_loop(self, connection: Connection):
        def prepare_sequence(empty_embedding: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [empty_embedding for _ in self._config.inference_sequence_length],
                dim=0,
            )

        def reset_sequence(sequence: torch.Tensor, empty_embedding: torch.Tensor):
            sequence[:, :] = empty_embedding.unsqueeze(0)


        reward_to_go = prepare_sequence(self._reward_empty_embedding)
        state_sequence = prepare_sequence(self._state_empty_embedding)
        action_sequence = prepare_sequence(self._action_empty_embedding)
        position_sequence = prepare_sequence(self._empty_positional_embedding)

        terminal = True
        first = True

        while not self._sigterm_detected.is_set():
            if not connection.poll(1.0):
                continue

            if terminal:
                reset_sequence(reward_to_go, self._reward_empty_embedding)
                reset_sequence(state_sequence, self._state_empty_embedding)
                reset_sequence(action_sequence, self._action_empty_embedding)
                reset_sequence(position_sequence, self._empty_positional_embedding)

    def _inference(self):
        self._sigterm_detected = threading.Event()

        conn, actor_conn = mp.Pipe(duplex=True)
        actor = Actor(self._environment, actor_conn, self._config.max_episode_steps)
        actor.run()
        try:
            self._inference_loop(conn)
        finally:
            actor.terminate()
            actor.join(30)
            if actor.exitcode is None:
                actor.kill()

    def run(self):
        return super().run()


class Actor(mp.Process):
    def __init__(
        self,
        environment: environments.Factory,
        inference_connection: Connection,
        max_environment_steps: int,
    ):
        super().__init__(daemon=True)
        self._stop_signal = None
        self._environment = environment
        self._inference_connection = inference_connection
        self._max_environment_steps = max_environment_steps

    def _sigterm(self, *args, **kwargs):
        self._stop_signal.set()

    def run(self) -> None:
        self._stop_signal = threading.Event()
        signal.signal(signal.SIGTERM, self._sigterm)

        env = self._environment()
        action_space = env.action_space.as_discrete()

        state, terminal, t, reward = None, True, 0, 0
        while not self._stop_signal.is_set():

            if terminal:
                state = env.reset()
            self._inference_connection.send(
                (
                    torch.as_tensor(state, dtype=torch.float16),
                    torch.as_tensor(action_space.action_mask, dtype=torch.bool),
                    reward,
                    terminal,
                )
            )
            terminal = False

            while not self._inference_connection.poll(1.0):
                if self._stop_signal.is_set():
                    break

            action: int = self._inference_connection.recv()
            state, reward, terminal, _ = env.step(action)
