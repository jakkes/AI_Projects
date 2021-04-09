import copy
from typing import Union

import numpy as np
from numpy import ndarray

import torch
from torch import nn, optim, Tensor

import ai.agents.utils.buffers as buffers
from ._agent_config import AgentConfig


def _apply_masks(values: Tensor, masks: Tensor) -> Tensor:
    return torch.where(masks, values, torch.empty_like(values).fill_(-np.inf))


def _get_actions(
    states: Tensor,
    action_masks: Tensor,
    network: nn.Module,
    use_distributional: bool,
    z: Tensor,
) -> Tensor:
    if use_distributional:
        d = network(states)
        values = torch.sum(d * z.view(1, 1, -1), dim=2)
    else:
        values = network(states).argmax(dim=1)
    values = _apply_masks(values, action_masks)
    return values.argmax(dim=1)


class Agent:
    """RainbowDQN Agent."""

    def __init__(
        self,
        config: AgentConfig,
        network: nn.Module,
        optimizer: optim.Optimizer = None,
        inference_mode: bool = False,
    ):
        """
        Args:
            config (AgentConfig): Agent configuration.
            network (nn.Module): Network.
            optimizer (optim.Optimizer): Optimizer.
            inference_mode (bool, optional): If True, the agent can only be used for
                acting. Saves memory by not initializing a replay buffer. Defaults to
                False.
        """
        self._network = network
        self._target_network = copy.deepcopy(network)
        self._optimizer = optimizer
        self._buffer = None
        if not inference_mode:
            self._initialize_not_inference_mode(config)

        if config.use_distributional:
            self._z = torch.linspace(config.v_min, config.v_max, steps=config.n_atoms)
            self._dz = self._z[1] - self._z[0]

        self.discount_factor = config.discount_factor
        """Discount factor used during training."""

        self._batch_vec = torch.arange(config.batch_size)
        if config.use_prioritized_experience_replay:
            self._beta_coeff = (config.beta_end - config.beta_start) / (
                config.beta_t_end - config.beta_t_start
            )

        self._train_steps = 0
        self._max_error = torch.tensor(1.0)
        self._config = config

    def _initialize_not_inference_mode(self, config: AgentConfig):
        if self._optimizer is None:
            raise ValueError(
                "Optimizer cannot be of NoneType when not running in inference mode."
            )

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
                config.replay_capacity, config.alpha, shapes, dtypes
            )
        else:
            self._buffer = buffers.Uniform(config.replay_capacity, shapes, dtypes)

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
                next_states,
                next_action_masks,
                self._network if self._config.use_double else self._target_network,
                self._config.use_distributional,
                self._z,
            )
        next_distribution = target_distribution[self._batch_vec, next_greedy_actions]

        m = torch.zeros(self._config.batch_size, self._config.n_atoms)
        projection = (
            rewards.view(-1, 1)
            + ~terminals.view(-1, 1) * self.discount_factor * self._z.view(1, -1)
        ).clamp_(self._config.v_min, self._config.v_max)
        b = (projection - self._config.v_min) / self._dz

        lower = b.floor().to(torch.long)
        upper = b.ceil().to(torch.long)
        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self._config.n_atoms - 1)) * (lower == upper)] += 1

        for batch in range(self._config.batch_size):
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
                    next_states,
                    next_action_masks,
                    self._network,
                    self._config.use_distributional,
                    self._z,
                )
                target_values = self._target_network(next_states)[
                    self._batch_vec, next_greedy_actions
                ]
        else:
            target_values = self._target_network(next_states).max(dim=1).values
        return torch.pow(
            rewards
            + ~terminals * self._config.discount_factor * target_values
            - current_q_values,
            2,
        )

    @property
    def config(self) -> AgentConfig:
        """Agent configuration in use."""
        return self._config

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
        errors = torch.as_tensor(errors, dtype=torch.float32)
        errors[errors.isnan()] = self._max_error
        self._buffer.add(
            (
                torch.as_tensor(states, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.long),
                torch.as_tensor(rewards, dtype=torch.float32),
                torch.as_tensor(terminals, dtype=torch.bool),
                torch.as_tensor(next_states, dtype=torch.float32),
                torch.as_tensor(next_action_masks, dtype=torch.bool),
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
            torch.as_tensor(state, dtype=torch.float32).unsqueeze_(0),
            torch.tensor([action], dtype=torch.long),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([terminal], dtype=torch.bool),
            torch.as_tensor(next_state, dtype=torch.float32).unsqueeze_(0),
            torch.as_tensor(next_action_mask, dtype=torch.bool).unsqueeze_(0),
            torch.tensor([error], dtype=torch.float32),
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
                torch.as_tensor(states, dtype=torch.float32),
                torch.as_tensor(action_masks, dtype=torch.bool),
                self._network,
                self._config.use_distributional,
                self._z,
            )

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
            torch.as_tensor(state, dtype=torch.float32).unsqueeze_(0),
            torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze_(0),
        )[0]

    def train_step(self):
        """Executes one training step."""

        (
            data,
            sample_probs,
            sample_ids,
        ) = self._buffer.sample(self._config.batch_size)

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
                (w * loss).mean().backward()
            else:
                updated_weights = loss.detach().pow(0.5)
                (w * loss).mean().backward()
            self._buffer.update_weights(sample_ids, updated_weights)
            self._max_error += 0.05 * (updated_weights.max() - self._max_error)
        else:
            loss.mean().backward()
        self._optimizer.step()

        self._train_steps += 1
        if self._train_steps % self._config.target_update_steps == 0:
            self._target_update()

    def buffer_size(self) -> int:
        """
        Returns:
            int: The current size of the replay buffer.
        """
        return self._buffer.size
