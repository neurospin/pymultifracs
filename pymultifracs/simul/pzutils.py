"""
Authors: Roberto Fabio Leonarduzzi
January, 2019
Copyright all rights reserved

Various helper functions for the synthesis of random processes.
"""


import numpy as np
import warnings
from scipy import linalg as spla


def gaussian_cme(cov, N, R, z=None):
    '''
    Create R realizations of a gaussian process of length N with the specified
    autocovariance through circulant matrix embedding.
    '''

    # Circulant matrix embedding: fft of periodized autocovariance:
    cov = np.concatenate((cov, np.flip(cov[1:-1])), axis=0)
    L = np.fft.fft(cov)[:, None]
    if np.any(np.real(L) < 0):
        warnings.warn('Found FFT of covariance < 0. Embedding matrix is not '
                      ' non-negative definite.')

    # Random noise in Fourier domain
    if z is None:
        z = np.random.randn(2*N - 2, R) + 1j * np.random.randn(2*N - 2, R)

    # Impose covariance and invert
    # Use fft to ignore normalization, because only real part is needed.
    x = np.real(np.fft.fft(z * np.sqrt(L / (2*N - 2)), axis=0))

    # First N samples have autocovariance cov:
    x = x[:N, :]

    return x


def gaussian_chol(cov, N, R, z=None):
    '''
    Create R realizations of a gaussian process of length N with the specified
    autocovariance through a Cholesky decomposition.

    Note the memory use is quadratic in N.

    Inputs:
      - cov: autocovariance sequence
      - N: sequence length
      - R: number of realizations
      - z (optional): initial white noise
    '''
    if z is None:
        z = np.random.randn(N, R)
    L = spla.cholesky(spla.toeplitz(cov))
    x = L @ z
    return x
