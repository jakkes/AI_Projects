from typing import Tuple
import threading
import struct

import torch
import zmq

import ai.rl.utils.seed as seed
import ai.rl.decision_transformer as dt
import ai.environments as environments

from ._actor import Actor


class RTGServer(threading.Thread):
    def __init__(self, dealer_address: str, exploration_strategy: dt.exploration_strategies.Base):
        super().__init__(name="RTGServer", daemon=True)
        self._exploration_strategy = exploration_strategy
        self._address = dealer_address

    def run(self):
        rep = zmq.Context.instance().socket(zmq.REP)
        rep.connect(self._address)

        while True:
            if rep.poll(timeout=5.0, flags=zmq.POLLIN) != zmq.POLLIN:
                continue

            rep.recv()
            rep.send(struct.pack("f", self._exploration_strategy.reward_to_go()))


def inference_shapes(config: dt.TrainerConfig) -> Tuple[Tuple[int, ...], ...]:
    return (
        (config.inference_sequence_length, ) + config.state_shape,
        (config.inference_sequence_length - 1, ) + config.action_shape,
        (config.inference_sequence_length, ),
        (config.inference_sequence_length, ),
        ()
    )


def inference_dtypes(config: dt.TrainerConfig) -> Tuple[torch.dtype, ...]:
    return (
        torch.float32,
        torch.long if config.discrete_action_space else torch.float32,
        torch.float32,
        torch.long,
        torch.long
    )



class Trainer:
    def __init__(
        self,
        agent: dt.Agent,
        config: dt.TrainerConfig,
        env: environments.Factory,
        exploration_strategy: dt.exploration_strategies.Base,
    ):
        self._agent = agent
        self._config = config
        self._env = env
        self._exploration_strategy = exploration_strategy

    def train(self, duration: float):
        """Trains the agent for the specified duration. This method blocks until
        training has finished.

        Args:
            duration (float): Duration in seconds.
        """
        proxy = seed.InferenceProxy()
        proxy_ports = proxy.start()

        rtgproxy = seed.InferenceProxy()
        rtgproxy_ports = rtgproxy.start()

        broadcaster = seed.Broadcaster(self._agent.model)
        broadcaster_port = broadcaster.start()

        inference_servers = [
            seed.InferenceServer(
                self._agent.model_factory,
                inference_shapes(self._config),
                inference_shapes(self._config),
                f"tcp://127.0.0.1:{proxy_ports[1]}",
                f"tcp://127.0.0.1:{broadcaster_port}",
                self._config.inference_batchsize,
                self._config.inference_delay,
                self._config.inference_device,
                daemon=True,
            )
            for _ in range(self._config.inference_servers)
        ]
        for server in inference_servers:
            server.start()

        actor_servers = [Actor()]
