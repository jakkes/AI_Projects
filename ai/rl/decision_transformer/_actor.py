import io
import struct
import threading
import multiprocessing as mp
from typing import List

import zmq
import torch

import ai.utils.logging as logging
import ai.rl.utils.seed as seed
import ai.rl.utils.buffers as buffers
import ai.environments as environments
import ai.rl.decision_transformer as dt


# Rounds action to the closest valid action.
def apply_action_mask(action: torch.Tensor, action_mask: torch.Tensor):
    action_vec = torch.arange(action_mask.shape[0])
    diff = (action_vec - action.unsqueeze(0)).abs_()[action_mask]
    action_vec = action_vec[action_mask]
    return action_vec[diff.argmin()]


def inference_data_store(config: dt.TrainerConfig) -> buffers.Uniform:
    return buffers.Uniform(
        config.inference_sequence_length,
        (config.state_shape, config.action_shape, ()),
        (
            torch.float32,
            torch.long if config.discrete_action_space else torch.float32,
            torch.float32,
        ),
    )


def sequence_data_store(config: dt.TrainerConfig) -> buffers.Uniform:
    return buffers.Uniform(
        config.max_environment_steps,
        (config.state_shape, config.action_shape, ()),
        (
            torch.float32,
            torch.long if config.discrete_action_space else torch.float32,
            torch.float32,
        ),
    )


class RTGClient:

    __slots__ = "_req"

    def __init__(self, rtg_request_address: str):
        self._req = zmq.Context.instance().socket(zmq.REQ)
        self._req.connect(rtg_request_address)

    def get(self) -> float:
        self._req.send(b"0")
        if self._req.poll(timeout=10000, flags=zmq.POLLIN) != zmq.POLLIN:
            raise RuntimeError("Failed fetching RTG.")

        return struct.unpack("f", self._req.recv())[0]


class ActorThread(threading.Thread):
    def __init__(
        self,
        config: dt.TrainerConfig,
        env: environments.Factory,
        inference_address: str,
        rtg_request_address: str,
        data_pub_address: str,
        log_client: logging.Client,
    ):
        super().__init__(daemon=False, name="ActorThread")
        self._env = env
        self._config = config
        self._inference_address = inference_address
        self._rtg_request_address = rtg_request_address
        self._data_pub_address = data_pub_address
        self._log_client = log_client

    def run(self):
        seed_client = seed.InferenceClient(self._inference_address)
        rtg_client = RTGClient(self._rtg_request_address)
        data_pub = seed.DataPublisher(self._data_pub_address)

        env = self._env()
        K = self._config.inference_sequence_length

        state = None
        terminal = True
        rtg = 0.0
        length = 0
        action = None
        step = 0

        states = torch.zeros(
            self._config.max_environment_steps, *self._config.state_shape
        )
        actions = torch.zeros(
            self._config.max_environment_steps,
            *self._config.action_shape,
            dtype=torch.float32
        )
        rtgs = torch.zeros(self._config.max_environment_steps)
        time_steps = torch.arange(self._config.max_environment_steps).float()

        while True:

            if terminal:
                if step > 0:
                    rtgs[:step] += reward - rtgs[step - 1]
                    data_pub.publish(
                        states, actions, rtgs, time_steps, torch.tensor(step)
                    )
                    self._log_client.log("Reward", rtgs[0].item())

                rtg = rtg_client.get()
                self._log_client.log("Start RTG", rtg)
                length = 0
                step = 0
                state = env.reset()
                terminal = False
                action = torch.as_tensor(
                    env.action_space.sample(),
                    dtype=torch.float32,
                )
                states.fill_(0)
                actions.fill_(0)
                rtgs.fill_(0)

            length = min(length + 1, K)
            states[step] = torch.as_tensor(state, dtype=torch.float32)
            rtgs[step] = rtg
            step += 1

            offset = max(0, K - step)
            action_output = seed_client.evaluate_model(
                states[step - length : step + offset],
                actions[step - length : step - 1 + offset],
                rtgs[step - length : step + offset],
                time_steps[step - length : step + offset],
                torch.tensor(length)
            )
            if self._config.discrete_action_space:
                action = int(
                    apply_action_mask(
                        action_output, env.action_space.as_discrete().action_mask
                    )
                )
            else:
                action = action_output
            actions[step - 1] = action

            next_state, reward, terminal, _ = env.step(action)
            if step >= self._config.max_environment_steps:
                terminal = True

            rtg -= reward
            state = next_state


class Actor(mp.Process):
    def __init__(
        self,
        config: dt.TrainerConfig,
        env: environments.Factory,
        inference_address: str,
        rtg_request_address: str,
        data_pub_address: str,
        log_client: logging.Client,
        daemon: bool = True,
    ):
        super().__init__(daemon=daemon, name="ActorProcess")
        self._env = env
        self._inference_address = inference_address
        self._config = config
        self._rtg_request_address = rtg_request_address
        self._data_pub_address = data_pub_address
        self._log_client = log_client
        self._threads: List[ActorThread] = []

    def run(self):
        self._threads = [
            ActorThread(
                self._config,
                self._env,
                self._inference_address,
                self._rtg_request_address,
                self._data_pub_address,
                self._log_client.clone(),
            )
            for _ in range(self._config.actor_threads)
        ]

        for thread in self._threads:
            thread.start()
