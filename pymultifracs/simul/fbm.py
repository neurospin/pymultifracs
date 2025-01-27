"""
Authors: Roberto Fabio Leonarduzzi
January, 2019

Synthesis of fractional brownian motions through circulant matrix embedding.
"""

import numpy as np
from .pzutils import gaussian_cme, gaussian_chol


def fgn(shape, H, sigma=1, dt=None, method='cme', z0=None):
    """
    Create a realization of fractional Brownian motion using circulant
    matrix embedding.

    Parameters
    ----------
    shape : int | tuple(int)
        If int, number of samples N. If tuple it is (N, R), the number of
        samples and realizations, respectively.
    H : float
        Hurst exponent.
    sigma : float
        Variance of process.

    Returns
    -------
    fbm : ndarray
        Synthesized fbm realizations. If `shape` is int, fbm is of shape
        (N,). Otherwise, it is of shape (N, R).
    """

    try:
        N, R = shape
        do_squeeze = False
    except TypeError:  # shape is scalar
        N, R = shape, 1
        do_squeeze = True

    if not 0 <= H <= 1:
        raise ValueError('H must satisfy 0 <= H <=1')

    if not dt:
        dt = 1 / N

    # Create covariance of fGn
    n = np.arange(N)
    r = dt**(2*H) * sigma**2 / 2 * (np.abs(n+1)**(2*H) + np.abs(n-1)**(2*H)
                                    - 2 * np.abs(n)**(2*H))
    # # Circulant matrix embedding: fft of periodized autocovariance:
    # r = np.concatenate((r, np.flip(r[1:-1])), axis=0)
    # L = np.fft.fft(r)[:, None]
    # if np.any(np.real(L) < 0):
    #     warnings.warn('Found FFT of covariance < 0. '
    #                   'Embedding matrix is not non-negative definite.')

    # # Random noise in Fourier domain
    # z = np.random.randn(2*N - 2, R) + 1j * np.random.randn(2*N - 2, R)

    # # Impose covariance and invert
    # # Use fft to ignore normalization, because only real part is needed.
    # x = np.real(np.fft.fft(z * np.sqrt(L) / (2*N - 2), axis=0))

    # # First N samples have the correct covariance:
    # fbm = np.cumsum(x[:N, :], axis=0)

    if method == 'cme':
        fGn = gaussian_cme(r, N, R)
    elif method == 'chol':
        fGn = gaussian_chol(r, N, R, z0)
    else:
        raise ValueError('Unknown synthesis method')

    return fGn.squeeze() if do_squeeze else fGn


def fbm(*args, **kwargs):
    """
    Simulate fBm.
    """
    return np.cumsum(fgn(*args, **kwargs), axis=0)
