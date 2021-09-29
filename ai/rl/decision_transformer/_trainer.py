import time
import signal
import threading
import queue
from typing import List, Sequence, Tuple
from multiprocessing.connection import Connection

import torch
from torch import nn, optim, multiprocessing as mp

from numpy import inf

import ai
from ai import environments
from ai.utils import logging
from . import exploration_strategies
from ._trainer_config import TrainerConfig


REPLAY_FEEDER_STEPS = 100


@torch.jit.script
def _to_batches(embeddings: torch.Tensor, lengths: torch.Tensor, inference_sequence_length: int):
    
    sequences = []

    for i, l in enumerate(lengths):
        for j in range(int(l)):
            sequences.append(embeddings[i, 3*j:3*(j+inference_sequence_length)-1])

    return torch.stack(sequences, dim=0)

@torch.jit.script
def _actions_to_batches(actions: torch.Tensor, lengths: torch.Tensor):
    out_actions = []
    for i, l in enumerate(lengths):
        out_actions.append(actions[i, :l])
    return torch.cat(out_actions)


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
        self._optimizer = optimizer
        self._config = config
        self._exploration_strategy = exploration_strategy
        self._environment = environment

        self._logging_queue = mp.Queue(maxsize=2000)
        self._logging_server = logging.SummaryWriterServer("DecisionTransformer", self._logging_queue)

        self._stop_training = threading.Event()

    def stop(self):
        """Signals for training to stop."""
        self._stop_training.set()

    def train(self):
        """Starts training. Blocks until training is terminated."""
        # inference =Inference(self._state_encoder, self._action_encoder, self._reward_encoder, self._positional_encoder)
        self._logging_server.start()
        
        data_queue = queue.Queue(10000)

        inference = Inference(
            self._state_encoder,
            self._action_encoder,
            self._reward_encoder,
            self._positional_encoder,
            self._action_decoder,
            self._transformer,
            self._state_empty_embedding,
            self._action_empty_embedding,
            self._reward_empty_embedding,
            self._empty_positional_embedding,
            self._environment,
            self._exploration_strategy,
            self._config,
            data_queue,
            self._logging_queue
        )
        inference.start()

        training = Training(
            self._state_encoder,
            self._action_encoder,
            self._reward_encoder,
            self._positional_encoder,
            self._action_decoder,
            self._transformer,
            self._state_empty_embedding,
            self._action_empty_embedding,
            self._reward_empty_embedding,
            self._empty_positional_embedding,
            self._environment,
            self._exploration_strategy,
            self._config,
            data_queue,
            self._optimizer,
            self._logging_queue
        )
        training.start()

        start_time = time.perf_counter()
        while (
            time.perf_counter() - start_time < self._config.training_time
            and not self._stop_training.is_set()
        ):
            time.sleep(1.0)

        inference.stop()
        training.stop()
        self._logging_server.terminate()

        inference.join()
        training.join()
        self._logging_server.join()



class Training(threading.Thread):
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
        optimizer: optim.Optimizer,
        logging_queue: queue.Queue,
    ):
        super().__init__(daemon=True)
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
        self._optimizer = optimizer

        self._sigterm_detected: threading.Event = threading.Event()
        self._train_steps: int = 0
        self._replay = ai.rl.utils.buffers.Uniform(
            self._config.replay_capacity,
            (
                (self._config.max_episode_steps,) + self._config.state_shape,  # State
                (
                    self._config.max_episode_steps,
                    self._config.action_size,
                ),  # Action mask
                (self._config.max_episode_steps,),  # Action
                (self._config.max_episode_steps,),  # Reward
                (),  # Sequence length
            ),
            (self._dtype, torch.bool, torch.long, self._dtype, torch.long),
        )
        self._replay_lock = threading.Lock()
        self._replay_feeder_thread: threading.Thread = None
        self._loss_fn = nn.CrossEntropyLoss()
        self._logging_queue = logging_queue

    def _replay_feeder(self):
        data = [
            torch.zeros(
                REPLAY_FEEDER_STEPS,
                self._config.max_episode_steps,
                *self._config.state_shape,
                dtype=self._dtype,
            ),
            torch.zeros(
                REPLAY_FEEDER_STEPS,
                self._config.max_episode_steps,
                self._config.action_size,
                dtype=torch.bool,
            ),
            torch.zeros(REPLAY_FEEDER_STEPS, self._config.max_episode_steps, dtype=torch.long),
            torch.zeros(REPLAY_FEEDER_STEPS, self._config.max_episode_steps, dtype=self._dtype),
            torch.zeros(REPLAY_FEEDER_STEPS, dtype=torch.long),
        ]
        i = 0
        while not self._sigterm_detected.is_set():
            try:
                sequence = self._data_queue.get(timeout=1.0)
                length = sequence[0].shape[0]
                for element, storage in zip(sequence, data):
                    storage[i, :length] = element
                data[-1][i] = length
                i += 1
            except queue.Empty:
                continue

            if i < REPLAY_FEEDER_STEPS:
                continue

            with self._replay_lock:
                self._replay.add(data, None)

            self._exploration_strategy.update(data[3])
            data = [torch.zeros_like(x) for x in data]

            if self._logging_queue is not None:
                self._logging_queue.put(logging.items.Scalar("Replay/Size", self._replay.size))

            i = 0

    def _block_until_large_enough_buffer(self):
        while not self._sigterm_detected.wait(1.0):
            if self._replay.size >= self._config.min_replay_size:
                return

    def _train_step(self):
        with self._replay_lock:
            (
                (states, action_masks, actions, rewards, lengths),
                _,
                _,
            ) = self._replay.sample(self._config.batch_size)
        embeddings = self._embed_sequences(states, actions, rewards)
        embeddings = self._pad_embeddings(*embeddings)
        embeddings = self._combine_embeddings(*embeddings)
        embeddings = _to_batches(embeddings, lengths, self._config.inference_sequence_length)
        target_actions = _actions_to_batches(actions.to(self._device), lengths)
        action_masks = _actions_to_batches(action_masks.to(self._device), lengths)
        action_logits = self._get_action_logits(embeddings, action_masks)

        loss = self._loss_fn(action_logits, target_actions)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        if self._logging_queue is not None:
            self._logging_queue.put(logging.items.Scalar("Trainer/Loss", loss.item()))

    def _combine_embeddings(self, states, actions, rewards, positions):
        embeddings = torch.stack(
            (rewards, states, actions), dim=2
        )
        embeddings += positions.unsqueeze(2)
        # first sequence is (should be) an empty action embedding
        return embeddings.view(embeddings.shape[0], -1, embeddings.shape[-1])

    def _embed_sequences(self, states, actions, rewards):
        states = self._state_encoder(states.to(self._device))
        actions = self._action_encoder(actions.to(self._device).to(self._dtype))
        rewards: torch.Tensor = rewards.to(self._device)
        rewards_to_go = rewards.sum(-1, keepdim=True).expand_as(rewards).clone()
        rewards_to_go[..., 1:] -= rewards[..., 1:]
        rewards_to_go = self._reward_encoder(rewards_to_go)
        positions = self._positional_encoder(torch.arange(self._config.max_episode_steps, device=self._device, dtype=self._dtype))
        return states, actions, rewards_to_go, positions

    def _get_action_logits(self, embeddings, action_masks):
        action_logits: torch.Tensor = self._action_decoder(
            self._transformer(embeddings)[:, -1]
        )
        action_logits[~action_masks] = -inf
        return action_logits

    def _pad_embeddings(self, states, actions, rewards, positions):
        def _pad(embeddings, empty_embedding, offset=1):
            return torch.cat(
                (
                    empty_embedding.view(1, 1, -1).expand(
                        embeddings.shape[0], self._config.inference_sequence_length - offset, -1
                    ),
                    embeddings,
                ),
                dim=1,
            )

        states = _pad(states, self._state_empty_embedding)
        actions = _pad(actions, self._action_empty_embedding)
        rewards = _pad(rewards, self._reward_empty_embedding)
        positions = _pad(positions.unsqueeze(0), self._empty_positional_embedding)

        return states, actions, rewards, positions

    def _trainer(self):
        self._block_until_large_enough_buffer()
        while not self._sigterm_detected.is_set():
            self._train_step()

    def stop(self):
        self._sigterm_detected.set()

    def run(self) -> None:
        self._replay_feeder_thread = threading.Thread(target=self._replay_feeder)
        self._replay_feeder_thread.start()

        self._trainer()

        self._replay_feeder_thread.join()


class Inference(threading.Thread):
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
        logging_queue: queue.Queue
    ):
        super().__init__(daemon=True)

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
        self._sigterm_detected: threading.Event = threading.Event()
        self._add_lock = threading.Lock()

        self._states: List[torch.Tensor] = []
        self._action_masks: List[torch.Tensor] = []
        self._actions: List[torch.Tensor] = []
        self._rewards_to_go: List[torch.Tensor] = []
        self._positions: List[torch.Tensor] = []
        self._results: List[int] = []
        self._executed_condition = threading.Condition(threading.Lock())

        self._logging_queue = logging_queue

    @property
    def data_queue(self) -> mp.Queue:
        return self._data_queue

    @property
    def exploration_strategy(self) -> exploration_strategies.Base:
        return self._exploration_strategy

    def stop(self):
        self._sigterm_detected.set()

    def _inference(self):
        conn, actor_conn = mp.Pipe(duplex=True)
        actor = Actor(
            self._environment, actor_conn, self._config.max_episode_steps, self._dtype
        )
        actor.start()
        try:
            InferenceLoop(conn, self, self._sigterm_detected, self._dtype, self._logging_queue).run()
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

        while not self._sigterm_detected.wait(5.0):
            pass

    def _reset_inference_data(self):
        self._states = []
        self._action_masks = []
        self._actions = []
        self._rewards_to_go = []
        self._positions = []
        self._results: List[int] = []
        self._executed_condition = threading.Condition(threading.Lock())

    def _embed_sequences(self):
        def embed(
            data: Sequence[torch.Tensor],
            lengths_cumsummed_start_0: torch.Tensor,
            encoder: nn.Module,
            offset=0,
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            embeddings = encoder(torch.cat(data, dim=0).to(self._dtype))
            return [
                embeddings[
                    lengths_cumsummed_start_0[i]
                    + offset * i : lengths_cumsummed_start_0[i + 1]
                    + offset * (i + 1)
                ]
                for i in range(lengths_cumsummed_start_0.shape[0] - 1)
            ]

        lengths = torch.tensor([x.shape[0] for x in self._states])
        lengths = torch.cat((torch.tensor([0]), lengths.cumsum(0)))

        state_embeddings = embed(self._states, lengths, self._state_encoder)
        action_embeddings = embed(
            self._actions, lengths, self._action_encoder, offset=-1
        )
        reward_embeddings = embed(self._rewards_to_go, lengths, self._reward_encoder)
        positional_embeddings = embed(
            self._positions, lengths, self._positional_encoder
        )

        return (
            state_embeddings,
            action_embeddings,
            reward_embeddings,
            positional_embeddings,
        )

    def _pad_embeddings(
        self,
        state_embeddings: List[torch.Tensor],
        action_embeddings: List[torch.Tensor],
        reward_embeddings: List[torch.Tensor],
        positional_embeddings: List[torch.Tensor],
    ):
        def pad(
            empty_embedding: torch.Tensor, embeddings: List[torch.Tensor]
        ) -> torch.Tensor:
            for i in range(len(embeddings)):
                npad = self._config.inference_sequence_length - embeddings[i].shape[0]
                if npad < 0:
                    raise RuntimeError()

                embeddings[i] = torch.cat(
                    (empty_embedding.unsqueeze(0).expand((npad, -1)), embeddings[i])
                )
            return torch.stack(embeddings)

        state_embeddings = pad(self._state_empty_embedding, state_embeddings)
        action_embeddings = pad(self._action_empty_embedding, action_embeddings)
        reward_embeddings = pad(self._reward_empty_embedding, reward_embeddings)
        positional_embeddings = pad(
            self._empty_positional_embedding, positional_embeddings
        )

        return (
            state_embeddings,
            action_embeddings,
            reward_embeddings,
            positional_embeddings,
        )

    def _combine_embeddings(
        self,
        state_embeddings,
        action_embeddings,
        reward_embeddings,
        positional_embeddings,
    ):
        embeddings = torch.stack(
            (action_embeddings, state_embeddings, reward_embeddings), dim=2
        )
        embeddings += positional_embeddings.unsqueeze(2)
        # first sequence is (should be) an empty action embedding
        return embeddings.view(embeddings.shape[0], -1, embeddings.shape[-1])[:, 1:]

    def _get_actions(self, embeddings):
        action_logits: torch.Tensor = self._action_decoder(
            self._transformer(embeddings)[:, -1]
        )
        action_masks = torch.stack(self._action_masks)
        action_logits[~action_masks] = -inf

        # Greedy action
        return action_logits.argmax(1)

    def _cut_sequences_and_move_to_device(self):
        def cut(sequence: List[torch.Tensor]):
            for i in range(len(sequence)):
                sequence[i] = sequence[i][-self._config.inference_sequence_length :].to(
                    self._device
                )

        cut(self._states)
        cut(self._actions)
        cut(self._rewards_to_go)
        cut(self._positions)

    def _execute(self):
        results = self._results
        condition = self._executed_condition

        with torch.no_grad():
            self._cut_sequences_and_move_to_device()
            actions = self._get_actions(
                self._combine_embeddings(
                    *self._pad_embeddings(
                        *self._embed_sequences(),
                    ),
                ),
            )
        self._reset_inference_data()

        for action in actions:
            results.append(action.item())
        with condition:
            condition.notify_all()

    def get_action(
        self,
        states: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
        reward_to_gos: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> int:
        with self._add_lock:
            self._states.append(states)
            self._action_masks.append(action_mask)
            self._actions.append(actions)
            self._rewards_to_go.append(reward_to_gos)
            self._positions.append(time_steps)
            i = len(self._states) - 1
            results = self._results
            condition = self._executed_condition

            if len(self._states) >= self._config.inference_batchsize:
                self._execute()

        with condition:
            condition.wait_for(lambda: len(results) > i)
            return results[i]


class InferenceLoop:
    def __init__(
        self,
        connection: Connection,
        inference: Inference,
        sigterm_detection: threading.Event,
        dtype: torch.dtype,
        logging_queue: queue.Queue
    ):
        self._conn = connection
        self._inference = inference
        self._sigterm_detection = sigterm_detection
        self._dtype = dtype

        self._terminal = True
        self._first = True

        self._states = []
        self._actions = []
        self._action_mask = None
        self._rewards = []
        self._time_steps = []
        self._time_step = 0
        self._reward_to_gos = [0.0]

        self._logging_queue = logging_queue

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
        self._action_mask = action_mask
        self._time_steps.append(self._time_step)
        self._time_step += 1

    def _handle_terminal(self):
        if not self._first:
            self._inference.data_queue.put(
                (
                    torch.stack(self._states),
                    self._action_mask,
                    torch.tensor(self._actions),
                    torch.tensor(self._rewards, dtype=self._dtype),
                )
            )
            if self._logging_queue is not None:
                self._logging_queue.put(logging.items.Scalar("Environment/Reward", sum(self._rewards)))
        else:
            self._first = False

        self._states = []
        self._action_mask = None
        self._actions = []
        self._rewards = []
        self._time_steps = []
        self._time_step = 0
        self._reward_to_gos = [self._inference.exploration_strategy.reward_to_go()]

        if self._logging_queue is not None:
            self._logging_queue.put(logging.items.Scalar("Environment/RewardToGo", self._reward_to_gos[0]))

    def _handle_data(self):
        state, action_mask, reward, self._terminal = self._conn.recv()

        self._finalize_step(reward)
        if self._terminal:
            self._handle_terminal()
        self._prepare_step(state, action_mask)
        self._execute_action(self._get_action())

    def _get_action(self):
        action = self._inference.get_action(
            torch.stack(self._states),
            self._action_mask,
            torch.tensor(self._actions),
            torch.tensor(self._reward_to_gos, dtype=self._dtype),
            torch.tensor(self._time_steps),
        )
        weights = torch.zeros(self._action_mask.shape)
        weights[self._action_mask] = 1.0
        random_action = ai.utils.torch.random.choice(weights)

        return random_action if torch.rand(1) < 0.1 else action

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
