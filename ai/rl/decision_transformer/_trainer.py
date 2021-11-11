from typing import Tuple
import time
import threading
import struct

import torch
import zmq

from ai.utils import Factory, logging
import ai.rl.utils.seed as seed
import ai.rl.decision_transformer as dt
import ai.environments as environments

from ._actor import Actor


class RTGServer(threading.Thread):
    def __init__(
        self, dealer_address: str, exploration_strategy: dt.exploration_strategies.Base
    ):
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
        (config.inference_sequence_length,) + config.state_shape,
        (config.inference_sequence_length - 1,) + config.action_shape,
        (config.inference_sequence_length,),
        (config.inference_sequence_length,),
        (),
    )


def inference_dtypes(config: dt.TrainerConfig) -> Tuple[torch.dtype, ...]:
    return (
        torch.float32,
        torch.long if config.discrete_action_space else torch.float32,
        torch.float32,
        torch.float32,
        torch.long,
    )


class Trainer:
    def __init__(
        self,
        agent: dt.Agent,
        optimizer: Factory[torch.optim.Optimizer],
        config: dt.TrainerConfig,
        env: environments.Factory,
        exploration_strategy: dt.exploration_strategies.Base,
    ):
        self._agent = agent
        self._config = config
        self._env = env
        self._exploration_strategy = exploration_strategy
        self._optimizer = optimizer

    def _train_loop(self, data: seed.DataCollector, log_client: logging.Client):
        while data.size < self._config.min_replay_size:
            time.sleep(1.0)

        print("Training started.")

        agent = self._agent
        batchsize = self._config.batch_size
        optimizer = self._optimizer(agent.model.parameters())
        loss_fn = torch.nn.MSELoss()
        while True:
            data, _, _ = data.sample(batchsize)
            rtgs = data[2]
            self._exploration_strategy.update(rtgs[:, 0], sequence=False)

            loss = agent.loss(*data, loss_fn=loss_fn)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            log_client.log("Loss", loss.item())
            log_client.log("Training rate", 1)
            

    def train(self, duration: float):
        """Trains the agent for the specified duration. This method blocks until
        training has finished.

        Args:
            duration (float): Duration in seconds.
        """
        log_server = logging.Server(
            logging.field.Scalar("Loss"),
            logging.field.Scalar("Reward"),
            logging.field.Scalar("Start RTG"),
            logging.field.Frequency("Data rate", 10),
            logging.field.Frequency("Training rate", 10),
            name="decisiontransformer"
        )
        log_port = log_server.start()
        log_client = logging.Client("127.0.0.1", log_port)

        proxy = seed.InferenceProxy()
        proxy_ports = proxy.start()

        rtgproxy = seed.InferenceProxy()
        rtgproxy_ports = rtgproxy.start()

        rtgserver = RTGServer(f"tcp://127.0.0.1:{rtgproxy_ports[1]}", self._exploration_strategy)
        rtgserver.start()

        broadcaster = seed.Broadcaster(self._agent.model, self._config.broadcast_period)
        broadcaster_port = broadcaster.start()


        data_collector = seed.DataCollector(
            self._config.replay_capacity,
            (
                (self._config.max_environment_steps,) + self._config.state_shape,
                (self._config.max_environment_steps,) + self._config.action_shape,
                (self._config.max_environment_steps,),
                (self._config.max_environment_steps,),
                (),
            ),
            inference_dtypes(self._config),
            device=self._config.replay_device,
            log_client=log_client.clone()
        )
        data_port = data_collector.start()

        inference_servers = [
            seed.InferenceServer(
                self._agent.model_factory,
                inference_shapes(self._config),
                inference_dtypes(self._config),
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

        actor_servers = [
            Actor(
                self._config,
                self._env,
                f"tcp://127.0.0.1:{proxy_ports[0]}",
                f"tcp://127.0.0.1:{rtgproxy_ports[0]}",
                f"tcp://127.0.0.1:{data_port}",
                log_client
            )
            for _ in range(self._config.actor_processes)
        ]
        for actor in actor_servers:
            actor.start()

        train_thread = threading.Thread(target=self._train_loop, args=(data_collector, log_client.clone()), daemon=True)
        train_thread.start()

        start_time = time.perf_counter()
        while time.perf_counter() - start_time < duration:
            time.sleep(1.0)

        print("Stopping...")
