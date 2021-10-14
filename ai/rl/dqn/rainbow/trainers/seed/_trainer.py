from typing import List
import random
import threading

import numpy as np
import zmq
import torch

import ai.rl as rl
import ai.rl.dqn.rainbow as rainbow
import ai.environments as environments
import ai.rl.utils.seed as seed
import ai.utils.logging as logging
from ._config import Config
from ._actor import Actor


def data_listener(self: "Trainer", data_port: int):
    pass


def create_server(
    self: "Trainer", dealer_port: int, broadcast_port: int
) -> seed.InferenceServer:
    return seed.InferenceServer(
        self._agent.model,
        self._agent.config.state_shape,
        torch.float32,
        f"tcp://127.0.0.1:{dealer_port}",
        f"tcp://127.0.0.1:{broadcast_port}",
        self._config.inference_batchsize,
        self._config.inference_delay,
        self._agent.config.network_device,
    )


def create_actor(self: "Trainer", data_port: int, router_port: int) -> Actor:
    return Actor(
        self._agent.inference_mode(),
        self._config,
        self._environment,
        data_port,
        router_port,
    )


class Trainer:
    """SEED trainer."""

    def __init__(
        self, agent: rainbow.Agent, config: Config, environment: environments.Factory
    ):
        self._agent = agent
        self._config = config
        self._environment = environment

        self._actors: List[Actor] = []
        self._proxy = seed.InferenceProxy()
        self._servers: List[seed.InferenceServer] = []
        self._broadcaster: seed.Broadcaster = seed.Broadcaster(agent.model, 2.5)
        self._data_sub: zmq.Socket = zmq.Context.instance().get(zmq.SUB)
        self._data_listening_thread: threading.Thread = None

    def start(self, duration: float):
        """Starts training, and blocks until completed.

        Args:
            duration (float): Training duration in seconds.
        """

        router_port, dealer_port = self._proxy.start()
        self._data_sub.subscribe("")
        data_port = self._data_sub.bind_to_random_port("tcp://*")
        broadcast_port = self._broadcaster.start()

        for _ in range(self._config.inference_servers):
            self._servers.append(create_server(self, dealer_port, broadcast_port))
        for server in self._servers:
            server.start()

        for _ in range(self._config.actor_processes):
            self._actors.append(create_actor(self, data_port, router_port))
        for actor in self._actors:
            actor.start()

        self._data_listening_thread = threading.Thread(
            target=data_listener, args=(data_port, ), daemon=True
        )
        self._data_listening_thread.start()
