from math import sqrt
from typing import Any, Callable, Mapping, Tuple

import torch
from torch import nn, optim
from torch.multiprocessing import Process
from torch.optim.optimizer import Optimizer

import ai
import ai.rl as rl
import ai.rl.a3c.trainer as trainer
import ai.environments as environments
import ai.utils.logging as logging


def _get_action_logit_value(
    network: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    state: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[int, float, torch.Tensor]:
    logits, v = network(state.unsqueeze(0))
    logits = logits.squeeze_(0)
    logits = torch.where(mask, logits, torch.zeros_like(logits) - float("inf"))
    action = ai.utils.torch.random.choice(torch.softmax(logits, dim=0)).item()
    return int(action), logits, v[0, 0]


class Worker(Process):
    def __init__(
        self,
        config: trainer.Config,
        environment: environments.Factory,
        network: nn.Module,
        optimizer_class: optim.Optimizer,
        optimizer_params: Mapping[str, Any],
        log_port: int,
    ):
        super().__init__(daemon=True)
        self._config = config
        self._network = network
        self._optimizer: Optimizer = None
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params
        self._environment = environment
        self._log_client = logging.Client("127.0.0.1", log_port)

        self._optimizer: Optimizer = None
        self._reward_collector: rl.utils.NStepRewardCollector = None
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
            self._state = torch.as_tensor(
                self._env.reset(), dtype=self._config.state_dtype
            )
            self._terminal = False
            with torch.no_grad():
                _, v = self._network(self._state.unsqueeze(0))
            self._log_client.log("Episode/Reward", self._episodic_reward)
            self._log_client.log("Episode/Start value", v.item())
            self._episodic_reward = 0.0

    def _add_loss(self, reward, terminal, logit, logits, value):
        stepinfo = self._reward_collector.step(reward, terminal, (logit, logits, value))
        if stepinfo is not None:
            (logits, logitss, values), rewards, terminals, (_, _, nvalues) = stepinfo
            self._steps += rewards.shape[0]
            advantage = (
                rewards + self._discount * ~terminals * nvalues.detach() - values
            )
            self._loss += (
                advantage.pow(2).sum()
                - (
                    advantage.detach()
                    * (logits - torch.softmax(logitss, dim=1).sum(1).log_())
                ).sum()
            )

            if self._steps >= self._config.batch_size:
                self._loss /= self._steps
                self._optimizer.zero_grad()
                self._loss.backward()
                gradmagn = sqrt(
                    sum(
                        param.grad.pow(2).sum().item()
                        for _, param in self._network.named_parameters()
                    )
                )
                self._optimizer.step()
                self._steps = 0
                self._loss = 0.0
                self._reward_collector.clear()

                self._log_client.log("Agent/Loss", self._loss.detach().item())
                self._log_client.log("Agent/Grad. magn.", gradmagn)

    def _step(self):
        action, logits, value = _get_action_logit_value(
            self._network, self._state, self._mask
        )
        next_state, reward, terminal, _ = self._env.step(action)
        next_state = torch.as_tensor(next_state, dtype=self._config.state_dtype)
        self._add_loss(reward, terminal, logits[action], logits, value)

        self._episodic_reward += reward
        self._state = next_state
        self._terminal = terminal

    def run(self) -> None:
        self._optimizer: optim.Optimizer = self._optimizer_class(
            self._network.parameters(), **self._optimizer_params
        )
        self._env = self._environment()
        self._action_space = self._env.action_space.as_discrete()
        self._reward_collector = rl.utils.NStepRewardCollector(
            self._config.n_step,
            self._config.discount,
            ((), (self._action_space.size,), ()),
            (torch.float32, torch.float32, torch.float32),
        )

        while True:
            self._check_reset()
            self._step()
