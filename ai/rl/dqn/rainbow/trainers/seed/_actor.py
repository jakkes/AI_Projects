import time
import multiprocessing as mp
import threading
from typing import List

import zmq

import ai.rl.dqn.rainbow as rainbow
import ai.environments as environments
import ai.rl.utils.seed as seed
from ai.rl.utils import NStepRewardCollector
from ._config import Config


class ActorThread(threading.Thread):
    def __init__(
        self, agent: rainbow.Agent, config: Config, environment: environments.Factory, router_port: int, data_port: int
    ):
        super().__init__(daemon=True)
        self._agent = agent
        self._environment = environment
        self._config = config
        self._data_port = data_port
        self._router_port = router_port
        self._client: seed.InferenceClient = None
        self._data_pub: zmq.Socket = None
        self._env: environments.Base = None

    def run(self):
        self._data_pub = zmq.Context.instance().socket(zmq.PUB)
        self._data_pub.connect(f"tcp://127.0.0.1:{self._data_port}")
        self._client = seed.InferenceClient(f"tcp://127.0.0.1:{self._router_port}")
        self._env = self._environment()


class Actor(mp.Process):
    def __init__(
        self,
        agent: rainbow.Agent,
        config: Config,
        environment: environments.Factory,
        data_port: int, router_port: int,
        daemon: bool = True,
    ):
        super().__init__(daemon=daemon)
        self._args = (agent, config, environment, router_port, data_port)

        self._threads: List[ActorThread] = []

    def run(self):
        for _ in range(self._config.actor_threads):
            self._threads.append(ActorThread(*self._args))
        for thread in self._threads:
            thread.start()

        while True:
            time.sleep(5.0)
