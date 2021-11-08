import struct
import threading
import multiprocessing as mp
from typing import List, Tuple

import zmq
import torch

import ai.rl.utils.seed as seed
import ai.rl.utils.buffers as buffers
import ai.environments as environments
import ai.rl.decision_transformer as dt


def inference_data_store(config: dt.TrainerConfig) -> buffers.Uniform:
    return buffers.Uniform(
        config.inference_sequence_length,
        (config.state_shape, config.action_shape, (), ()),
        (
            torch.float32,
            torch.long if config.discrete_action_space else torch.float32,
            torch.long,
            torch.long,
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
    ):
        super().__init__(daemon=False, name="ActorThread")
        self._env = env
        self._config = config
        self._inference_address = inference_address
        self._rtg_request_address = rtg_request_address

    def run(self):
        seed_client = seed.InferenceClient(self._inference_address)
        rtg_client = RTGClient(self._rtg_request_address)
        buffer = inference_data_store(self._config)
        env = self._env()

        state = None
        terminal = True
        rtg = 0.0
        length = 0
        action = None

        while True:

            if terminal:
                buffer.clear()
                rtg = rtg_client.get()
                length = 0
                state = env.reset()
                terminal = False
                action = env.action_space.sample()

            length = min(length + 1, self._config.inference_sequence_length)
            state = torch.as_tensor(state, dtype=torch.float32)



class Actor(mp.Process):
    def __init__(
        self,
        config: dt.TrainerConfig,
        env: environments.Factory,
        inference_address: str,
        rtg_request_address: str,
        daemon: bool = True,
    ):
        super().__init__(daemon=daemon, name="ActorProcess")
        self._env = env
        self._inference_address = inference_address
        self._config = config
        self._rtg_request_address = rtg_request_address
        self._threads: List[ActorThread] = []

    def run(self):
        self._threads = [
            ActorThread(
                self._config,
                self._env,
                self._inference_address,
                self._rtg_request_address,
            )
            for _ in range(self._config.actor_threads)
        ]

        for thread in self._threads:
            thread.start()
