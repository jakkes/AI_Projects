from multiprocessing import Queue
from rl.utils.logging import SummaryWriterServer


class Implementation(SummaryWriterServer):
    def log(self, writer, data):
        pass


def test_start_term_join():
    server = Implementation("test", Queue())
    server.start()
    server.terminate()
    server.join()
