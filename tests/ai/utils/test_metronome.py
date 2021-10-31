import time
import random
from ai.utils import Metronome


def test_metronome():
    sync = Metronome(0.5)

    sync.wait()

    last = time.perf_counter()
    for _ in range(10):
        time.sleep(random.random() * 0.5)
        sync.wait()
        assert (time.perf_counter() - last - 0.5) < 0.05

        last = time.perf_counter()
