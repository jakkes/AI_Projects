import torch

import ai.agents as agents
import ai.environments as environments
from ... import Agent
from ._config import Config


class Trainer:
    """Trainer class."""

    def __init__(self, agent: Agent, config: Config, environment: environments.Factory):
        """
        Args:
            agent (Agent): Agent.
            config (Config): Trainer configuration.
            environment (env.Factory): Environment factory.
        """
        self._agent = agent
        self._config = config
        self._env_factory = environment

        self._reward_collector = agents.utils.NStepRewardCollector(
            config.n_step,
            agent.discount_factor,
            (agent.config.state_shape, (), (agent.config.action_space_size,)),
            (torch.float32, torch.long, torch.bool),
        )

        agent.discount_factor = agent.discount_factor ** config.n_step

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

        while not terminal:
            step += 1
            action = self._agent.act_single(state, mask)
            next_state, reward, terminal, _ = env.step(action)

            if (
                self._config.max_environment_steps > 0
                and step >= self._config.max_environment_steps
            ):
                terminal = True

            self._add_to_collector(state, action, mask, reward, terminal)
            state = next_state
            mask = env.action_space.as_discrete().action_mask

            self._train_step()

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
            states, actions, rewards, terminals, next_states, next_action_masks
        )

    def start(self):
        """Starts training, according to the configuration."""

        env = self._env_factory()
        try:
            self._run(env)
        finally:
            env.close()
