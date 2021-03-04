from multiprocessing import set_start_method, get_start_method
from . import environments, simulators, utils, agents

__all__ = ["environments", "simulators", "utils", "agents"]
__version__ = "0.0.3"

try:
    set_start_method("spawn")
except RuntimeError:
    if get_start_method().lower() != "spawn":
        raise RuntimeError("Start method is not spawn")
