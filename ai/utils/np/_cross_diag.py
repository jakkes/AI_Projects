import numpy as np


def cross_diag(values: np.ndarray) -> np.ndarray:
    """Returns a matrix with `values` placed on the cross diagonal.

    Args:
        values (np.ndarray): Values

    Returns:
        np.ndarray: Matrix with `values` on the cross diagonal. Shape is `(n, n)`, where
        `n` is the length of `values`.
    """
    n = values.shape[0]
    i = np.arange(n)
    re = np.zeros((n, n), dtype=values.dtype)
    re[i, -i - 1] = values
    return re
