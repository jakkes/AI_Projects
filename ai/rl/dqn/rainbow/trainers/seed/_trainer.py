import copy
import time
import io
import threading

import numpy as np
import zmq
import torch

import ai.rl.dqn.rainbow as rainbow
import ai.environments as environments
import ai.rl.utils.seed as seed
import ai.utils.logging as logging
from ai.utils import Metronome
from ._config import Config
from ._actor import Actor


def data_listener(
    agent: rainbow.Agent,
    data_sub: zmq.Socket,
    logger_port: int,
    stop_event: threading.Thread,
):
    logger = logging.Client("127.0.0.1", logger_port)

    steps = 0

    while not stop_event.is_set():
        if data_sub.poll(timeout=1000, flags=zmq.POLLIN) != zmq.POLLIN:
            continue
        data = torch.load(io.BytesIO(data_sub.recv()))
        agent.observe(
            data[0][0], data[0][1], data[1], data[2], data[3][0], data[3][2], np.nan
        )

        steps += data[0][0].shape[0]
        if steps >= 100:
            logger.log("Buffer/Size", agent.buffer_size())
            logger.log("Buffer/Data freq.", steps)
            steps = 0


def trainer(
    agent: rainbow.Agent, config: Config, stop_event: threading.Event, logging_port: int
):
    logger = logging.Client("localhost", logging_port)

    while agent.buffer_size() < config.minimum_buffer_size and not stop_event.is_set():
        time.sleep(1.0)

    agent.discount_factor = agent.discount_factor ** config.n_step

    metronome = None
    if config.max_train_frequency > 0:
        metronome = Metronome(1.0 / config.max_train_frequency)

    while not stop_event.is_set():
        if metronome is not None:
            metronome.wait()

        agent.train_step()
        logger.log("Trainer/Train freq.", 1)


def create_server(
    self: "Trainer", dealer_port: int, broadcast_port: int
) -> seed.InferenceServer:
    return seed.InferenceServer(
        self._agent.model_factory,
        (self._agent.config.state_shape, ),
        (torch.float32, ),
        f"tcp://127.0.0.1:{dealer_port}",
        f"tcp://127.0.0.1:{broadcast_port}",
        self._config.inference_batchsize,
        self._config.inference_delay,
        self._agent.config.network_device,
    )


def create_logger() -> logging.Server:
    return logging.Server(
        logging.field.Scalar("Environment/Reward"),
        logging.field.Scalar("Buffer/Size"),
        logging.field.Scalar("RainbowAgent/Loss"),
        logging.field.Scalar("RainbowAgent/Max error"),
        logging.field.Scalar("RainbowAgent/Gradient norm"),
        logging.field.Scalar("Actor/Start value"),
        logging.field.Frequency("Trainer/Train freq.", 5.0),
        logging.field.Frequency("Buffer/Data freq.", 5.0),
        name="dqnseed",
    )


def create_actor(
    self: "Trainer", data_port: int, router_port: int, logger_port: int
) -> Actor:
    return Actor(
        self._agent.config,
        self._config,
        self._environment,
        data_port,
        router_port,
        logging_client=logging.Client("127.0.0.1", logger_port),
        daemon=False
    )


class Trainer:
    """SEED trainer."""

    def __init__(
        self, agent: rainbow.Agent, config: Config, environment: environments.Factory
    ):
        self._agent = agent
        self._config = config
        self._environment = environment

    def start(self, duration: float):
        """Starts training, and blocks until completed.

        Args:
            duration (float): Training duration in seconds.
        """
        proxy = seed.InferenceProxy()
        router_port, dealer_port = proxy.start()

        data_sub = zmq.Context.instance().socket(zmq.SUB)
        data_sub.subscribe("")
        data_port = data_sub.bind_to_random_port("tcp://*")

        broadcaster = seed.Broadcaster(self._agent.model_instance, self._config.broadcast_period)
        broadcast_port = broadcaster.start()

        logger = create_logger()
        logger_port = logger.start()

        self._agent.set_logging_client(logging.Client("localhost", logger_port))

        stop_event = threading.Event()

        data_listening_thread = threading.Thread(
            target=data_listener,
            args=(self._agent, data_sub, logger_port, stop_event),
            daemon=True,
        )
        data_listening_thread.start()

        servers = [
            create_server(self, dealer_port, broadcast_port)
            for _ in range(self._config.inference_servers)
        ]
        for server in servers:
            server.start()

        # Allow some time to start server.
        time.sleep(5.0)

        actors = [
            create_actor(self, data_port, router_port, logger_port)
            for _ in range(self._config.actor_processes)
        ]
        for actor in actors:
            actor.start()

        training_thread = threading.Thread(
            target=trainer,
            args=(self._agent, self._config, stop_event, logger_port),
            daemon=True,
        )
        training_thread.start()

        # Allow some time to start actors.
        time.sleep(5.0)

        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            time.sleep(5.0)

            for i in range(len(actors)):
                if not actors[i].is_alive():
                    actors[i] = create_actor(self, data_port, router_port, logger_port)
                    actors[i].start()
                    print("Restarted actor...")

        stop_event.set()

        for actor in actors:
            actor.terminate()
        for server in servers:
            server.terminate()
        for actor in actors:
            actor.join()
        for server in servers:
            server.join()
        data_sub.close()

        training_thread.join()
        data_listening_thread.join()
