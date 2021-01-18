"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
"""

import numpy as np


def generate_fbm_path(N, H, n_paths=1, end_time=10):
    """
    Generates n_paths paths of length N of a fractal Brownian motion of
    parameter H
    """

    time_vec = np.linspace(0, end_time, N)
    time_vec = time_vec[1:]
    N -= 1

    T, S = np.meshgrid(time_vec, time_vec)

    # Covariance matrix
    R = 0.5*(T**(2*H) + S**(2*H) - np.abs(T-S)**(2*H))

    X = generate_gaussian_vector(N, R, n_samples=n_paths)
    X = np.vstack((np.zeros((1, n_paths)), X))

    # Return path
    return X


def generate_gaussian_vector(N, C, mu=None, n_samples=1):
    """
    Generate n_samples of a gaussian vector of length N.
    Dimension of the output = (N, n_samples)

    Input C must be a (N, N) matrix
    Input mu must have N elements
    """
    if mu is None:
        mu = np.zeros((N, n_samples))
    else:
        mu = mu.reshape((N, 1))
        mu = mu.dot(np.ones((1, n_samples)))

    v = np.random.randn(N, n_samples)

    L = np.linalg.cholesky(C)
    u = L.dot(v) + mu
    return u
