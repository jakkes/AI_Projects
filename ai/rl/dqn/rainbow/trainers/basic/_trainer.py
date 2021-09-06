import random
import numpy as np
import torch
from multiprocessing import Queue

import ai.rl as rl
import ai.rl.dqn.rainbow as rainbow
import ai.environments as environments
from ._config import Config
from ._logger import Logger


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        agent: rainbow.Agent,
        config: Config,
        environment: environments.Factory,
    ):
        """
        Args:
            agent (rainbow.Agent): Agent.
            config (Config): Trainer configuration.
            environment (environments.Factory): Environment factory.
        """
        self._agent = agent
        self._config = config
        self._env_factory = environment

        self._logging_queue = Queue(maxsize=2000)
        self._logging_server = Logger(self._logging_queue)
        self._agent.set_logging_queue(self._logging_queue)

        self._reward_collector = rl.utils.NStepRewardCollector(
            config.n_step,
            agent.config.discount_factor,
            (agent.config.state_shape, (), (agent.config.action_space_size,)),
            (torch.float32, torch.long, torch.bool),
        )

        agent.discount_factor = agent.config.discount_factor ** config.n_step

    def _run(self, env: environments.Base):
        for _ in range(self._config.episodes):
            self._run_episode(env)

    def _train_step(self):
        if self._agent.buffer_size() > self._config.minimum_buffer_size:
            self._agent.train_step()

    def _run_episode(self, env: environments.Base):
        state = env.reset()
        mask = env.action_space.as_discrete().action_mask
        terminal = False
        step = -1
        total_reward = 0.0
        total_discounted_reward = 0.0
        start_value = self._agent.q_values_single(state, mask).max().item()

        while not terminal:
            step += 1
            if random.random() < self._config.epsilon:
                action = env.action_space.sample()
            else:
                action = self._agent.act_single(state, mask)
            next_state, reward, terminal, _ = env.step(action)
            total_reward += reward
            total_discounted_reward = (
                reward + self._agent.config.discount_factor * total_discounted_reward
            )

            if (
                self._config.max_environment_steps > 0
                and step >= self._config.max_environment_steps
            ):
                terminal = True

            self._add_to_collector(state, action, mask, reward, terminal)
            state = next_state
            mask = env.action_space.as_discrete().action_mask

            self._train_step()

        self._logging_queue.put({
            "reward": total_reward,
            "discounted_reward": total_discounted_reward,
            "steps": step,
            "start_value": start_value
        })

    def _add_to_collector(self, state, action, action_mask, reward, terminal):
        out = self._reward_collector.step(
            reward, terminal, (state, action, action_mask)
        )
        if out is None:
            return

        (
            (states, actions, _),
            rewards,
            terminals,
            (next_states, _, next_action_masks),
        ) = out
        self._agent.observe(
            states, actions, rewards, terminals, next_states, next_action_masks, np.nan
        )

    def start(self):
        """Starts training, according to the configuration."""

        self._logging_server.start()

        env = self._env_factory()
        try:
            self._run(env)
        finally:
            env.close()
            self._logging_server.terminate()
            self._logging_server.join()