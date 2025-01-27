"""
Authors: Merlin Dumeur <merlin@dumeur.net>

Extends rupture package functionality for outlier detection.
"""

from numba import jit
import numpy as np
import ruptures as rpt


@jit(nopython=True)
def hilbert(a, b):
    """
    Modified Hilbert metric.
    """

    norm = np.linalg.norm

    return np.log(
        (1 + norm(b - a, ord=2) / norm(1 - a, ord=2))
        * (1 + norm(b-a, ord=2) / norm(1 - b, ord=2)))


@jit(nopython=True)
def w_hilbert(a, b, w):
    """
    Weighted version of the modified Hilbert metric.
    """

    w = np.sqrt(w)
    norm = np.linalg.norm
    a = a * w
    b = b * w
    one = w

    return np.log(
        (1 + norm(b - a, ord=2) / norm(one - a, ord=2))
        * (1 + norm(b-a, ord=2) / norm(one - b, ord=2)))


@jit(nopython=True)
def _numba_mean(X, axis=None):
    """
    Mean function for numba
    """

    if axis is None:
        return np.mean(X)

    return np.sum(X, axis=axis) / X.shape[axis]


@jit(nopython=True)
def _hilbert_cost(X, w):

    mu = np.exp(_numba_mean(np.log(X), axis=0))

    distance = np.zeros(X.shape[0])

    for i in range(distance.shape[0]):
        distance[i] = w_hilbert(X[i], mu, w)

    return np.sum(distance ** 2)


class HilbertCost(rpt.base.BaseCost):
    """
    Custom cost class for the ruptures package
    """
    model = ""
    min_size = 2

    def __init__(self, w=None):
        self.w = w

    def fit(self, signal):
        """
        Sets the signal.
        """

        self.signal = signal
        return self

    def error(self, start, end):
        """
        Compute modified Hilbert metric cost.
        """

        return _hilbert_cost(self.signal[start:end], self.w)
