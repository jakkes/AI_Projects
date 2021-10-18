import random
import io
import time
import multiprocessing as mp
import threading
from typing import List

import numpy as np
import zmq
import torch

import ai.utils.logging as logging
import ai.environments as environments
import ai.rl.dqn.rainbow as rainbow
import ai.rl.utils.seed as seed
from ai.rl.utils import NStepRewardCollector
from ._config import Config


def send_data(data, pub: zmq.Socket):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    pub.send(buffer.getvalue())


@torch.jit.script
def _apply_masks(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return torch.where(masks, values, torch.empty_like(values).fill_(-np.inf))


@torch.jit.script
def _get_actions(
    action_masks: torch.Tensor,
    network_output: torch.Tensor,
    use_distributional: bool,
    z: torch.Tensor,
) -> torch.Tensor:
    if use_distributional:
        d = network_output
        values = torch.sum(d * z.view(1, 1, -1), dim=2)
    else:
        values = network_output
    values = _apply_masks(values, action_masks)
    return values.argmax(dim=1)


class ActorThread(threading.Thread):
    def __init__(
        self,
        agent_config: rainbow.AgentConfig,
        config: Config,
        environment: environments.Factory,
        router_port: int,
        data_port: int,
        logging_client: logging.Client = None,
    ):
        super().__init__(daemon=True)
        self._agent_config = agent_config
        self._config = config
        self._environment = environment
        self._data_port = data_port
        self._router_port = router_port
        self._logging_client = logging_client

    def run(self):
        data_pub = zmq.Context.instance().socket(zmq.PUB)
        data_pub.connect(f"tcp://127.0.0.1:{self._data_port}")
        client = seed.InferenceClient(f"tcp://127.0.0.1:{self._router_port}")
        
        device = self._config.inference_device
        env = self._environment()
        action_space = env.action_space.as_discrete()
        reward_collector = NStepRewardCollector(
            self._config.n_step,
            self._agent_config.discount_factor,
            [
                self._agent_config.state_shape,
                (),
                (self._agent_config.action_space_size,),
            ],
            [torch.float32, torch.long, torch.bool],
            device=device
        )
        max_steps = self._config.max_environment_steps
        logging_client = self._logging_client
        z = torch.linspace(
            self._agent_config.v_min,
            self._agent_config.v_max,
            steps=self._agent_config.n_atoms,
            device=self._config.inference_device,
        )
        use_distributional = self._agent_config.use_distributional

        terminal = True
        state = None
        steps = None
        total_reward = None

        while True:
            if terminal:
                if total_reward is not None and logging_client is not None:
                    logging_client.log("Environment/Reward", total_reward)

                state = env.reset()
                steps = 0
                total_reward = 0
                terminal = False

            state = torch.as_tensor(state, dtype=torch.float32, device=device)
            mask = torch.as_tensor(action_space.action_mask, dtype=torch.bool, device=device)

            if random.random() < self._config.epsilon:
                action = action_space.sample()
            else:
                model_output = client.evaluate_model(state)
                action = _get_actions(mask.unsqueeze(0), model_output.unsqueeze(0), use_distributional, z)[0]

            next_state, reward, terminal, _ = env.step(action)
            total_reward += reward
            steps += 1

            if max_steps > 0 and steps >= max_steps:
                terminal = True

            data = reward_collector.step(reward, terminal, (state, action, mask))
            if data is not None:
                send_data(data, data_pub)

            state = next_state


class Actor(mp.Process):
    def __init__(
        self,
        agent_config: rainbow.AgentConfig,
        config: Config,
        environment: environments.Factory,
        data_port: int,
        router_port: int,
        logging_client: logging.Client = None,
        daemon: bool = True,
    ):
        super().__init__(daemon=daemon)
        self._config = config
        self._args = (agent_config, config, environment, router_port, data_port)
        self._kwargs = {"logging_client": logging_client}

        self._threads: List[ActorThread] = []

    def run(self):
        for _ in range(self._config.actor_threads):
            self._threads.append(ActorThread(*self._args, **self._kwargs))
        for thread in self._threads:
            thread.start()

        while True:
            time.sleep(5.0)
