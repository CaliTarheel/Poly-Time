"""Euclidean length of a single real vector, without np.linalg.norm's argument handling.

For a one-dimensional float64 array NumPy computes ``sqrt(x.dot(x))`` after a ravel that
is the identity. Doing exactly those two operations on the array gives a bitwise identical
result several times faster, which matters for the millions of three-vectors the spherical
geometry code normalizes. Anything else (other dtypes, ranks) takes NumPy's own path.
"""
import math
import numpy as np


def length(x):
    """float(np.linalg.norm(x)), bit for bit."""
    if type(x) is np.ndarray and x.ndim == 1 and x.dtype == np.float64:
        return math.sqrt(x.dot(x))
    return float(np.linalg.norm(x))
