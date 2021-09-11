from abc import abstractmethod, ABC
from queue import Empty
from multiprocessing import Process, Queue, Event
from torch.utils.tensorboard.writer import SummaryWriter

import ai.utils.logging.items as logitems


class SummaryWriterServer(ABC):
    """Process handling a summary writer."""

    def __init__(self, filename_suffix: str, data_queue: Queue):
        """
        Args:
            filename_suffix (str): Filename suffix of the SummaryWriter.
            data_queue (Queue): Queue from which logging data is gathered.
        """
        self._data_queue = data_queue
        self._filename_suffix = filename_suffix
        self._stop_signal = Event()
        self._process: Process = None

    def log(self, summary_writer: SummaryWriter, data: logitems.Base):
        """This method is called whenever data is retrieved from the data queue.

        Args:
            summary_writer (SummaryWriter): Summary writer used by this server.
            data (Any): Data, defined by the publishing entity.
        """
        data.log(summary_writer)

    def _runner(self):
        summary_writer = SummaryWriter(comment=self._filename_suffix)
        while not self._stop_signal.is_set():
            try:
                data = self._data_queue.get(timeout=5)
            except Empty:
                continue
            self.log(summary_writer, data)

    def start(self):
        """Starts the logging server."""
        self._process = Process(target=self._runner, daemon=True)
        self._process.start()

    def terminate(self):
        """Stops the logging server."""
        self._stop_signal.set()

    def join(self):
        """Blocks until the logging server has been stopped."""
        self._process.join()
