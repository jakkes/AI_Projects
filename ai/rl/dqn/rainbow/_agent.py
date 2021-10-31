import copy
from typing import Dict, Union

import numpy as np
from numpy import ndarray

import torch
from torch import nn, optim, Tensor

import ai.rl.utils.buffers as buffers
import ai.utils.logging as logging
from ai.utils import Factory
from ._agent_config import AgentConfig


@torch.jit.script
def _apply_masks(values: Tensor, masks: Tensor) -> Tensor:
    return torch.where(masks, values, torch.empty_like(values).fill_(-np.inf))


@torch.jit.script
def _get_actions(
    action_masks: Tensor,
    network_output: Tensor,
    use_distributional: bool,
    z: Tensor,
) -> Tensor:
    if use_distributional:
        d = network_output
        values = torch.sum(d * z.view(1, 1, -1), dim=2)
    else:
        values = network_output
    values = _apply_masks(values, action_masks)
    return values.argmax(dim=1)


@torch.jit.script
def _get_distributional_loss(
    rewards: Tensor,
    terminals: Tensor,
    current_distribution: Tensor,
    target_distribution: Tensor,
    next_actions: Tensor,
    batch_vector: Tensor,
    z: Tensor,
    dz: float,
    device: torch.device,
    batchsize: int,
    n_atoms: int,
    v_max: float,
    v_min: float,
    discount: float,
):
    next_distribution = target_distribution[batch_vector, next_actions]
    m = torch.zeros(batchsize, n_atoms, device=device)

    projection = (
        rewards.view(-1, 1) + ~terminals.view(-1, 1) * discount * z.view(1, -1)
    ).clamp_(v_min, v_max)
    b = (projection - v_min) / dz

    lower = b.floor().to(torch.long).clamp_(0, n_atoms - 1)
    upper = b.ceil().to(torch.long).clamp_(0, n_atoms - 1)
    lower[(upper > 0) * (lower == upper)] -= 1
    upper[(lower < (n_atoms - 1)) * (lower == upper)] += 1

    for batch in range(batchsize):
        m[batch].put_(
            lower[batch],
            next_distribution[batch] * (upper[batch] - b[batch]),
            accumulate=True,
        )
        m[batch].put_(
            upper[batch],
            next_distribution[batch] * (b[batch] - lower[batch]),
            accumulate=True,
        )
    return -(m * current_distribution.add_(1e-6).log_()).sum(1)


class Agent:
    """RainbowDQN Agent."""

    def __init__(
        self,
        config: AgentConfig,
        network: Factory[nn.Module],
        optimizer: Factory[optim.Optimizer] = None,
        inference_mode: bool = False,
        replay_init_lazily: bool = True
    ):
        """
        Args:
            config (AgentConfig): Agent configuration.
            network (Factory[nn.Module]): Network wrapped in a `Factory`.
            optimizer (Factory[optim.Optimizer]): Optimizer, wrapped in a `Factory`.
                Model parameters are passed to the optimizer when instanced.
            inference_mode (bool, optional): If `True`, the agent can only be used for
                acting. Saves memory by not initializing a replay buffer. Defaults to
                False.
            replay_init_lazily (bool, optional): If `True`, the replay buffer is
                initialized lazily, i.e. when the first observation is added. Defaults
                to `True`.
        """
        self._config = config
        self._network_factory = network
        self._network = network().to(config.network_device)
        self._target_network = network().to(config.network_device)
        self._optimizer = optimizer
        self._buffer = None
        self._td_loss = (
            torch.nn.HuberLoss(reduction="none")
            if config.huber_loss
            else torch.nn.MSELoss(reduction="none")
        )
        if not inference_mode:
            self._initialize_not_inference_mode(config, replay_init_lazily)

        self._z = torch.linspace(
            config.v_min,
            config.v_max,
            steps=config.n_atoms,
            device=self._config.network_device,
        )
        self._dz = self._z[1] - self._z[0]

        self.discount_factor = config.discount_factor
        """Discount factor used during training."""

        self._batch_vec = torch.arange(
            config.batch_size, device=self._config.network_device
        )
        if config.use_prioritized_experience_replay:
            self._beta_coeff = (config.beta_end - config.beta_start) / (
                config.beta_t_end - config.beta_t_start
            )

        self._train_steps = 0
        self._max_error = torch.tensor(1.0)
        self._logging_client: logging.Client = None

    def _initialize_replay(self, config: AgentConfig):
        shapes = (
            config.state_shape,  # state
            (),  # action
            (),  # reward
            (),  # terminal
            config.state_shape,  # next state
            (config.action_space_size,),  # next action mask
        )
        dtypes = (
            torch.float32,
            torch.long,
            torch.float32,
            torch.bool,
            torch.float32,
            torch.bool,
        )

        if config.use_prioritized_experience_replay:
            self._buffer = buffers.Weighted(
                config.replay_capacity,
                config.alpha,
                shapes,
                dtypes,
                self._config.replay_device,
            )
        else:
            self._buffer = buffers.Uniform(
                config.replay_capacity, shapes, dtypes, self._config.replay_device
            )

    def _initialize_not_inference_mode(self, config: AgentConfig, replay_init_lazily: bool):
        if self._optimizer is None:
            raise ValueError(
                "Optimizer cannot be `None` when not running in inference mode."
            )
        self._optimizer = self._optimizer(self._network.parameters())
        if not replay_init_lazily:
            self._initialize_replay(config)


    def _target_update(self):
        self._target_network.load_state_dict(self._network.state_dict())

    def _get_distributional_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        next_states: Tensor,
        next_action_masks: Tensor,
    ) -> Tensor:
        current_distribution: Tensor = self._network(states)[self._batch_vec, actions]
        target_distribution = self._target_network(next_states)
        with torch.no_grad():
            next_greedy_actions = _get_actions(
                next_action_masks,
                (self._network if self._config.use_double else self._target_network)(
                    next_states
                ),
                self._config.use_distributional,
                self._z,
            )
        return _get_distributional_loss(
            rewards,
            terminals,
            current_distribution,
            target_distribution,
            next_greedy_actions,
            self._batch_vec,
            self._z,
            self._dz,
            self._config.network_device,
            self._config.batch_size,
            self._config.n_atoms,
            self._config.v_max,
            self._config.v_min,
            self._config.discount_factor,
        )

    def _get_td_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        next_states: Tensor,
        next_action_masks: Tensor,
    ) -> Tensor:
        current_q_values = self._network(states)[self._batch_vec, actions]
        if self._config.use_double:
            with torch.no_grad():
                next_greedy_actions = _get_actions(
                    next_action_masks,
                    self._network(next_states),
                    self._config.use_distributional,
                    self._z,
                )
                target_values = self._target_network(next_states)[
                    self._batch_vec, next_greedy_actions
                ]
        else:
            target_values = self._target_network(next_states).max(dim=1).values
        return self._td_loss(
            current_q_values,
            rewards + ~terminals * self._config.discount_factor * target_values,
        )

    @property
    def config(self) -> AgentConfig:
        """Agent configuration in use."""
        return self._config

    @property
    def model_factory(self) -> Factory[nn.Module]:
        """Model factory used by the agent."""
        return self._network_factory

    @property
    def model_instance(self) -> nn.Module:
        """Model instance."""
        return self._network

    def _get_actions(self, action_masks: Tensor, network_output: Tensor):
        return _get_actions(
            action_masks, network_output, self._config.use_distributional, self._z
        )

    def observe(
        self,
        states: Union[Tensor, ndarray],
        actions: Union[Tensor, ndarray],
        rewards: Union[Tensor, ndarray],
        terminals: Union[Tensor, ndarray],
        next_states: Union[Tensor, ndarray],
        next_action_masks: Union[Tensor, ndarray],
        errors: Union[Tensor, ndarray],
    ):
        """Adds a batch of experiences to the replay

        Args:
            states (Union[Tensor, ndarray]): States
            actions (Union[Tensor, ndarray]): Actions
            rewards (Union[Tensor, ndarray]): Rewards
            terminals (Union[Tensor, ndarray]): Terminal flags
            next_states (Union[Tensor, ndarray]): Next states
            next_action_masks (Union[Tensor, ndarray]): Next action masks
            errors (Union[Tensor, ndarray]): TD errors. NaN values are replaced by
                appropriate initialization value.
        """
        if self._buffer is None:
            self._initialize_replay(self._config)

        errors = torch.as_tensor(errors, dtype=torch.float32)
        errors[errors.isnan()] = self._max_error
        self._buffer.add(
            (
                torch.as_tensor(
                    states, dtype=torch.float32, device=self._config.replay_device
                ),
                torch.as_tensor(
                    actions, dtype=torch.long, device=self._config.replay_device
                ),
                torch.as_tensor(
                    rewards, dtype=torch.float32, device=self._config.replay_device
                ),
                torch.as_tensor(
                    terminals, dtype=torch.bool, device=self._config.replay_device
                ),
                torch.as_tensor(
                    next_states, dtype=torch.float32, device=self._config.replay_device
                ),
                torch.as_tensor(
                    next_action_masks,
                    dtype=torch.bool,
                    device=self._config.replay_device,
                ),
            ),
            errors,
        )

    def observe_single(
        self,
        state: Union[Tensor, ndarray],
        action: int,
        reward: float,
        terminal: bool,
        next_state: Union[Tensor, ndarray],
        next_action_mask: Union[Tensor, ndarray],
        error: float,
    ):
        """Adds a single experience to the replay buffer.

        Args:
            state (Union[Tensor, ndarray]): State
            action (int): Action
            reward (float): Reward
            terminal (bool): True if `next_state` is a terminal state
            next_state (Union[Tensor, ndarray]): Next state
            next_action_mask (Union[Tensor, ndarray]): Next action mask
            error (float): TD error. NaN values are replaced by appropriate
                initialization value.
        """
        self.observe(
            torch.as_tensor(
                state, dtype=torch.float32, device=self._config.replay_device
            ).unsqueeze_(0),
            torch.tensor([action], dtype=torch.long, device=self._config.replay_device),
            torch.tensor(
                [reward], dtype=torch.float32, device=self._config.replay_device
            ),
            torch.tensor(
                [terminal], dtype=torch.bool, device=self._config.replay_device
            ),
            torch.as_tensor(
                next_state, dtype=torch.float32, device=self._config.replay_device
            ).unsqueeze_(0),
            torch.as_tensor(
                next_action_mask, dtype=torch.bool, device=self._config.replay_device
            ).unsqueeze_(0),
            torch.tensor(
                [error], dtype=torch.float32, device=self._config.replay_device
            ),
        )

    def act(
        self, states: Union[Tensor, ndarray], action_masks: Union[Tensor, ndarray]
    ) -> Tensor:
        """Returns the greedy action for the given states and action masks.

        Args:
            states (Union[Tensor, ndarray]): States.
            action_masks (Union[Tensor, ndarray]): Action masks.

        Returns:
            Tensor: Tensor of dtype `torch.long`.
        """
        with torch.no_grad():
            return _get_actions(
                torch.as_tensor(
                    action_masks, dtype=torch.bool, device=self._config.network_device
                ),
                self._network(
                    torch.as_tensor(
                        states, dtype=torch.float32, device=self._config.network_device
                    )
                ),
                self._config.use_distributional,
                self._z,
            ).cpu()

    def act_single(
        self, state: Union[Tensor, ndarray], action_mask: Union[Tensor, ndarray]
    ) -> int:
        """Returns the greedy action for one state-action mask pair.

        Args:
            state (Union[Tensor, ndarray]): State.
            action_mask (Union[Tensor, ndarray]): Action mask.

        Returns:
            int: Action index.
        """
        return self.act(
            torch.as_tensor(
                state, dtype=torch.float32, device=self._config.network_device
            ).unsqueeze_(0),
            torch.as_tensor(
                action_mask, dtype=torch.bool, device=self._config.network_device
            ).unsqueeze_(0),
        )[0]

    def train_step(self):
        """Executes one training step."""

        (
            data,
            sample_probs,
            sample_ids,
        ) = self._buffer.sample(self._config.batch_size)

        data = (x.to(self._config.network_device) for x in data)

        if self._config.use_distributional:
            loss = self._get_distributional_loss(*data)
        else:
            loss = self._get_td_loss(*data)

        self._optimizer.zero_grad()
        if self._config.use_prioritized_experience_replay:
            beta = min(
                max(
                    self._beta_coeff * (self._train_steps - self._config.beta_t_start)
                    + self._config.beta_start,
                    self._config.beta_start,
                ),
                self._config.beta_end,
            )
            w = (1.0 / self._buffer.size / sample_probs) ** beta
            w /= w.max()
            if self._config.use_distributional:
                updated_weights = loss.detach()
            else:
                updated_weights = loss.detach().pow(0.5)
            loss = (w * loss).mean()
            self._buffer.update_weights(sample_ids, updated_weights)
            self._max_error += 0.05 * (updated_weights.max() - self._max_error)
        else:
            loss = loss.mean()
        loss.backward()
        grad_norm = None
        if self.config.gradient_norm > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                self._network.parameters(), self.config.gradient_norm
            )
        self._optimizer.step()

        self._train_steps += 1
        if self._train_steps % self._config.target_update_steps == 0:
            self._target_update()

        if self._logging_client is not None:
            self._logging_client.log("RainbowAgent/Loss", loss.detach().item())
            self._logging_client.log("RainbowAgent/Max error", self._max_error.item())

            if grad_norm is not None:
                self._logging_client.log("RainbowAgent/Gradient norm", grad_norm.item())

    def buffer_size(self) -> int:
        """
        Returns:
            int: The current size of the replay buffer.
        """
        return 0 if self._buffer is None else self._buffer.size

    def set_logging_client(self, client: logging.Client):
        """Sets (and overrides previously set) logging client used by the agent. The
        agent outputs tensorboard logs through this client.

        Args:
            client (logging.Client): Client.
        """
        self._logging_client = client

    def q_values(
        self, states: Union[Tensor, ndarray], action_masks: Union[Tensor, ndarray]
    ) -> Tensor:
        """Computes the Q-values for each state-action pair. Illegal actions are given
        a values -inf.

        Args:
            states (Union[Tensor, ndarray]): States.
            action_masks (Union[Tensor, ndarray]): Action masks.

        Returns:
            Tensor: Q-values.
        """

        states = torch.as_tensor(
            states, dtype=torch.float32, device=self._config.network_device
        )
        action_masks = torch.as_tensor(
            action_masks, dtype=torch.bool, device=self._config.network_device
        )

        with torch.no_grad():
            if self.config.use_distributional:
                values = (self._z.view(1, 1, -1) * self._network(states)).sum(2)
            else:
                values = self._network(states)
            return _apply_masks(values, action_masks).cpu()

    def q_values_single(
        self, state: Union[Tensor, ndarray], action_mask: Union[Tensor, ndarray]
    ) -> Tensor:
        """Computes the Q-values of all state-action pairs, given a single state.

        Args:
            state (Union[Tensor, ndarray]): State.
            action_mask (Union[Tensor, ndarray]): Action mask.

        Returns:
            Tensor: Q-values, illegal actions have value -inf.
        """
        return self.q_values(
            torch.as_tensor(
                state, dtype=torch.float32, device=self._config.network_device
            ).unsqueeze_(0),
            torch.as_tensor(
                action_mask, dtype=torch.bool, device=self._config.network_device
            ).unsqueeze_(0),
        )[0]

    def inference_mode(self) -> "Agent":
        """Returns a copy of this agent, but in inference mode.
        
        Returns:
            Agent: A shallow copy of the agent, capable of inference only."""
        return Agent(self._config, self._network, inference_mode=True)
