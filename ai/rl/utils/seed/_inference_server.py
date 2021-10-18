import warnings
import io
import threading
from typing import List, Tuple

import zmq

import torch
from torch import nn, multiprocessing as mp

import ai.rl.utils.buffers as buffers


def param_listener(model: nn.Module, address: str):
    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.subscribe("")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(address)

    while True:
        if socket.poll(timeout=5000, flags=zmq.POLLIN) != zmq.POLLIN:
            continue
        data = io.BytesIO(socket.recv())
        model.load_state_dict(torch.load(data))


def create_buffer(self: "InferenceServer") -> buffers.Uniform:
    return buffers.Uniform(
        self._batchsize, (self._state_shape,), (self._state_dtype,), self._device
    )


def start_rep_workers(self: "InferenceServer"):
    self._rep_working_threads = [
        threading.Thread(target=rep_worker, args=(self,), daemon=True)
        for _ in range(self._batchsize)
    ]
    for worker in self._rep_working_threads:
        worker.start()


def rep_worker(self: "InferenceServer"):
    def get_output(data):
        for _ in range(10):
            try:
                return self._get_batch().evaluate_model(data)
            except Batch.Executed:
                continue
        raise RuntimeError("Failed evaluating data sample, ten attempts were made.")

    socket = zmq.Context.instance().socket(zmq.REP)
    socket.connect(self._dealer_address)

    while True:
        if socket.poll(timeout=1000, flags=zmq.POLLIN) != zmq.POLLIN:
            continue
        recvbytes = io.BytesIO(socket.recv())
        output = get_output(torch.load(recvbytes))
        data = io.BytesIO()
        torch.save(output, data)
        socket.send(data.getvalue())


class InferenceServer(mp.Process):
    """Process serving inference requests from clients."""

    def __init__(
        self,
        model: nn.Module,
        state_shape: Tuple[int, ...],
        state_dtype: torch.dtype,
        dealer_address: int,
        broadcast_address: str,
        batchsize: int,
        max_delay: float,
        device: torch.device = torch.device("cpu"),
        daemon: bool = True,
    ):
        """
        Args:
            model (nn.Module): Model served. __Note__: model must not be on GPU. If the
                model should be served on the GPU, pass a CPU version of the model and
                pass argument `device`.
            state_shape (Tuple[int, ...]): Shape of states, excluding batch size.
            state_dtype (torch.dtype): Data type of states.
            dealer_address (int): Address to `InferenceProxy` session, e.g.
            `tcp://127.0.0.1:33333`.
            broadcast_address (str): Address to `Broadcaster` session, e.g.
                `tcp://127.0.0.1:33332`.
            batchsize (int): Batch size of inference requests.
            max_delay (float): Maximum delay (seconds) before inference is executed
                regardless of batchsize.
            device (torch.device, optional): Device on which the model should be run on.
                Defaults to torch.device("cpu").
            daemon (bool, optional): Whether or not to run the process in daemon-mode.
                Defaults to True.
        """
        super().__init__(daemon=daemon)
        self._model = model
        self._state_shape = state_shape
        self._state_dtype = state_dtype
        self._dealer_address = dealer_address
        self._broadcast_address = broadcast_address
        self._batchsize = batchsize
        self._max_delay = max_delay
        self._device = device

        self._rep_working_threads: List[threading.Thread] = None
        self._batch: "Batch" = None
        self._batch_lock: threading.Lock = None

    def run(self):
        self._model = self._model.to(self._device)

        self._batch_lock = threading.Lock()
        start_rep_workers(self)
        param_listener(self._model, self._broadcast_address)

    def _get_batch(self) -> "Batch":
        with self._batch_lock:
            if self._batch is None or self._batch.executed:
                self._batch = Batch(self._model, create_buffer(self), self._max_delay)
            return self._batch


class Batch:
    class Executed(Exception):
        pass

    # __slots__ = (
    #     "_buffer",
    #     "_model",
    #     "_output",
    #     "_executed_condition",
    #     "_executed_event",
    #     "_timer",
    # )

    def __init__(self, model: nn.Module, buffer: buffers.Uniform, max_delay: float):
        self._buffer = buffer
        self._model = model
        self._output = None
        self._executed_condition = threading.Condition(threading.Lock())
        self._executed_event = threading.Event()
        self._timer = threading.Timer(interval=max_delay, function=self.execute)

    @property
    def executed(self) -> bool:
        return self._executed_event.is_set()

    def evaluate_model(self, data: torch.Tensor) -> torch.Tensor:
        with self._executed_condition:
            if self._executed_event.is_set():
                raise Batch.Executed

            label = self._buffer.add((data,), None, batch=False)

            if self._buffer.size >= self._buffer.capacity:
                self.execute(lock_acquired=True)
            elif not self._timer.is_alive():
                self._timer.start()

            success = self._executed_condition.wait_for(
                lambda: self._executed_event.is_set(), timeout=10.0
            )

        if not success:
            raise RuntimeError("Model execution timed out.")
        return self._output[label]

    def execute(self, lock_acquired: bool = False):
        if not lock_acquired or self._executed_event.is_set():
            with self._executed_condition:
                return self.execute(lock_acquired=True)

        if self._executed_event.is_set():
            return

        data, _, _ = self._buffer.get_all()
        with torch.no_grad():
            self._output = self._model(*data)
        self._executed_event.set()
        self._executed_condition.notify_all()
