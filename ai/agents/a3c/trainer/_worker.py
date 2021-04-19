from queue import Queue
from typing import Any, Callable, Mapping, Tuple

import torch
from torch import nn, optim
from torch.multiprocessing import Process
from torch.optim.optimizer import Optimizer

import ai
import ai.agents as agents
import ai.agents.a3c.trainer as trainer
import ai.environments as environments


def _get_action_logit_value(
    network: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    state: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[int, float, torch.Tensor]:
    p, v = network(state.unsqueeze(0))
    p = torch.where(mask, p, torch.zeros_like(p))
    action = ai.utils.torch.random.choice(p).item()
    return int(action), p[0, action], v[0, 0]


class Worker(Process):
    def __init__(
        self,
        config: trainer.Config,
        environment: environments.Factory,
        network: nn.Module,
        optimizer_class: optim.Optimizer,
        optimizer_params: Mapping[str, Any],
        logging_queue: Queue
    ):
        super().__init__(daemon=True)
        self._config = config
        self._network = network
        self._optimizer: Optimizer = None
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params
        self._environment = environment
        self._logging_queue = logging_queue

        self._optimizer: Optimizer = None
        self._reward_collector: agents.utils.NStepRewardCollector = None
        self._env: environments.Base = None
        self._action_space: environments.action_spaces.Discrete = None
        self._discount = self._config.discount ** self._config.n_step
        self._terminal = True
        self._state = None
        self._steps = 0
        self._loss = 0.0
        self._episodic_reward = 0.0

    @property
    def _mask(self):
        return torch.as_tensor(self._action_space.action_mask, dtype=torch.bool)

    def _check_reset(self):
        if self._terminal:
            self._state = torch.as_tensor(self._env.reset(), dtype=self._config.state_dtype)
            self._terminal = False
            self._logging_queue.put({"r": self._episodic_reward})
            self._episodic_reward = 0.0

    def _add_loss(self, reward, terminal, logit, value):
        stepinfo = self._reward_collector.step(reward, terminal, (logit, value))
        if stepinfo is not None:
            (logits, values), rewards, terminals, (_, nvalues) = stepinfo
            self._steps += rewards.shape[0]
            advantage = (
                rewards + self._discount * ~terminals * nvalues.detach() - values
            )
            self._loss += (
                advantage.pow(2).sum() - (advantage.detach() * logits.log()).sum()
            )

            if self._steps >= self._config.batch_size:
                self._loss /= self._steps
                self._logging_queue.put(self._loss.detach().item())
                self._optimizer.zero_grad()
                self._loss.backward()
                self._optimizer.step()
                self._steps = 0
                self._loss = 0.0
                self._reward_collector.clear()

    def _step(self):
        action, logit, value = _get_action_logit_value(self._network, self._state, self._mask)
        next_state, reward, terminal, _ = self._env.step(action)
        next_state = torch.as_tensor(next_state, dtype=self._config.state_dtype)
        self._add_loss(reward, terminal, logit, value)

        self._episodic_reward += reward
        self._state = next_state
        self._terminal = terminal

    def run(self) -> None:
        self._optimizer: optim.Optimizer = self._optimizer_class(
            self._network.parameters(), **self._optimizer_params
        )
        self._env = self._environment()
        self._action_space = self._env.action_space.as_discrete()
        self._reward_collector = agents.utils.NStepRewardCollector(
            self._config.n_step,
            self._config.discount,
            ((), ()),
            (torch.float32, torch.float32),
        )

        while True:
            self._check_reset()
            self._step()
