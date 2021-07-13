import signal
import threading
from typing import List, Sequence, Tuple
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
        action_decoder: nn.Module,
        state_empty_embedding: torch.Tensor,
        action_empty_embedding: torch.Tensor,
        reward_empty_embedding: torch.Tensor,
        position_empty_embedding: torch.Tensor,
        environment: environments.Factory,
        exploration_strategy: exploration_strategies.Base,
        config: TrainerConfig,
        optimizer: optim.Optimizer,
    ):
        self._device = torch.device("cuda" if config.enable_cuda else "cpu")
        self._dtype = torch.float16 if config.enable_float16 else torch.float32

        self._state_encoder = (
            state_encoder.to(self._dtype).to(self._device).share_memory()
        )
        self._action_encoder = (
            action_encoder.to(self._dtype).to(self._device).share_memory()
        )
        self._reward_encoder = (
            reward_encoder.to(self._dtype).to(self._device).share_memory()
        )
        self._positional_encoder = (
            positional_encoder.to(self._dtype).to(self._device).share_memory()
        )
        self._action_decoder = (
            action_decoder.to(self._dtype).to(self._device).share_memory()
        )
        self._transformer = transformer.to(self._dtype).to(self._device).share_memory()
        self._state_empty_embedding = (
            state_empty_embedding.to(self._dtype).to(self._device).share_memory_()
        )
        self._action_empty_embedding = (
            action_empty_embedding.to(self._dtype).to(self._device).share_memory_()
        )
        self._reward_empty_embedding = (
            reward_empty_embedding.to(self._dtype).to(self._device).share_memory_()
        )
        self._empty_positional_embedding = (
            position_empty_embedding.to(self._dtype).to(self._device).share_memory_()
        )
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
        action_decoder: nn.Module,
        transformer: nn.Module,
        state_empty_embedding: torch.Tensor,
        action_empty_embedding: torch.Tensor,
        reward_empty_embedding: torch.Tensor,
        position_empty_embedding: torch.Tensor,
        environment: environments.Factory,
        exploration_strategy: exploration_strategies.Base,
        config: TrainerConfig,
        data_queue: mp.Queue,
    ):
        super().__init__()

        self._device = torch.device("cuda" if config.enable_cuda else "cpu")
        self._dtype = torch.float16 if config.enable_float16 else torch.float32

        self._state_encoder = state_encoder
        self._action_encoder = action_encoder
        self._reward_encoder = reward_encoder
        self._positional_encoder = positional_encoder
        self._action_decoder = action_decoder
        self._transformer = transformer
        self._state_empty_embedding = state_empty_embedding
        self._action_empty_embedding = action_empty_embedding
        self._reward_empty_embedding = reward_empty_embedding
        self._empty_positional_embedding = position_empty_embedding
        self._config = config
        self._exploration_strategy = exploration_strategy
        self._environment = environment
        self._data_queue = data_queue

        self._inference_threads: List[threading.Thread] = []
        self._sigterm_detected: threading.Event = None

    @property
    def data_queue(self) -> mp.Queue:
        return self._data_queue

    @property
    def exploration_strategy(self) -> exploration_strategies.Base:
        return self._exploration_strategy

    def _sigterm(self, *args, **kwargs):
        self._sigterm_detected.set()

    def _inference(self):
        self._sigterm_detected = threading.Event()

        conn, actor_conn = mp.Pipe(duplex=True)
        actor = Actor(
            self._environment, actor_conn, self._config.max_episode_steps, self._dtype
        )
        actor.run()
        try:
            InferenceLoop(conn, self).run()
        finally:
            actor.terminate()
            actor.join(30)
            if actor.exitcode is None:
                actor.kill()

    def run(self):
        self._inference_threads = [
            threading.Thread(target=self._inference)
            for _ in range(self._config.number_of_actors)
        ]
        for thread in self._inference_threads:
            thread.start()

    def get_action(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        reward_to_gos: torch.Tensor,
    ) -> int:
        raise NotImplementedError


class InferenceLoop:
    def __init__(
        self,
        connection: Connection,
        inference: Inference,
        sigterm_detection: threading.Event,
        dtype: torch.dtype,
    ):
        self._conn = connection
        self._inference = inference
        self._sigterm_detection = sigterm_detection
        self._dtype = dtype

        self._terminal = True
        self._first = True

        self._states = []
        self._action_masks = []
        self._rewards = []
        self._reward_to_gos = [0.0]

    def run(self):
        while not self._sigterm_detection.is_set():
            if not self._conn.poll(1.0):
                continue
            self._handle_data()

    def _finalize_step(self, reward):
        self._reward_to_gos.append(self._reward_to_gos[-1] - reward)
        self._rewards.append(reward)

    def _prepare_step(self, state, action_mask):
        self._states.append(state)
        self._action_masks.append(action_mask)

    def _handle_terminal(self):
        if not self._first:
            self._inference.data_queue.put(
                (
                    torch.stack(self._states),
                    torch.stack(self._action_masks),
                    torch.tensor(self._actions),
                    torch.tensor(self._rewards, dtype=self._dtype),
                )
            )
        else:
            self._first = False

        self._states = []
        self._action_masks = []
        self._actions = []
        self._rewards = []
        self._reward_to_gos = [self._inference.exploration_strategy.reward_to_go()]

    def _handle_data(self):
        state, action_mask, reward, self._terminal = self._conn.recv()

        self._finalize_step(reward)
        if self._terminal:
            self._handle_terminal()
        self._prepare_step(state, action_mask)
        self._execute_action(self._get_action())

    def _get_action(self):
        return self._inference.get_action(
            torch.stack(self._states),
            torch.stack(self._action_masks),
            torch.tensor(self._reward_to_gos, dtype=self._dtype),
        )

    def _execute_action(self, action):
        self._actions.append(action)
        self._conn.send(action)


class Actor(mp.Process):
    def __init__(
        self,
        environment: environments.Factory,
        inference_connection: Connection,
        max_environment_steps: int,
        dtype: torch.dtype,
    ):
        super().__init__(daemon=True)
        self._stop_signal = None
        self._environment = environment
        self._inference_connection = inference_connection
        self._max_environment_steps = max_environment_steps
        self._dtype = dtype

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
                    torch.as_tensor(state, dtype=self._dtype),
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
