import time


class Metronome:
    """Utility for executing at a certain frequency."""
    
    def __init__(self, period: float):
        """
        Args:
            period (float): Period (seconds) that this timer should help control.
        """
        self._period = period
        self._last_return = None

    def wait(self):
        """Blocks until the specified period has passed since the last time this method
        returned."""
        if self._last_return is None:
            time.sleep(self._period)
        else:
            time.sleep(max(self._period - time.perf_counter() + self._last_return, 0))
        self._last_return = time.perf_counter()
