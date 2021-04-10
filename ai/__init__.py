import queue
import os
from types import ModuleType
from multiprocessing import set_start_method, get_start_method
from . import environments, simulators, utils, agents

__all__ = ["environments", "simulators", "utils", "agents"]
__version__ = "0.0.4"

try:
    set_start_method("spawn")
except RuntimeError:
    if get_start_method().lower() != "spawn":
        raise RuntimeError("Start method is not spawn")


root = os.path.dirname(__file__)

modules = queue.Queue()
for module in __all__:
    modules.put_nowait((module, ))

while not modules.empty():
    module_name = modules.get_nowait()
    module = eval(".".join(module_name))
    if not isinstance(module, ModuleType):
        continue
    if "__pdoc__" not in dir(module):
        module.__pdoc__ = {}
    if "__all__" not in dir(module):
        module.__all__ = []

    for submodule in os.listdir(os.path.join(root, *module_name)):
        submodule_path = os.path.join(root, *module_name, submodule)
        if submodule.startswith("_"):
            continue
        if os.path.isdir(submodule_path) and "__init__.py" in os.listdir(submodule_path):
            module.__pdoc__[submodule] = submodule in module.__all__
            if submodule in module.__all__:
                modules.put_nowait(module_name + (submodule, ))
        elif submodule.endswith(".py"):
            submodule = submodule[:-3]
            module.__pdoc__[submodule] = submodule in module.__all__
