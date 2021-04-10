import os
from typing import Dict, Sequence

from python_linq import From


def get_pdoc_dict(all: Sequence[str], init_fp: str) -> Dict[str, bool]:
    """Generates the `__pdoc__` dictionary of a module.

    Args:
        all (Sequence[str]): `__all__` variable defined in the module.
        init_fp (str): File path to the module's `__init__.py` file.

    Returns:
        Dict[str, bool]: `__pdoc__` dictionary.
    """
    root = os.path.dirname(init_fp)
    ls = os.listdir(root)

    re = {}

    for name in ls:
        if os.path.isdir(os.path.join(root, name)):
            re[name] = name in all
        elif name.endswith(".py"):
            name = name[:-3]
            re[name] = name in all
    return re