# Synthesis of multifractal random walk and derived processes.
#
# Roberto Fabio Leonarduzzi
# January, 2019

import numpy as np
from .fbm import fgn
from .pzutils import gaussian_cme, gaussian_chol
from numpy.fft import fft, ifft
# import math
# import matplotlib.pyplot as plt


def mrw(shape, H, lam, L, sigma=1, method='cme', z0=(None, None)):
    '''
    Create a realization of fractional Brownian motion using circulant
    matrix embedding.

    Parameters
    ----------
    shape : int | tuple(int)
        If scalar, it is the  number of samples. If tuple it is (N, R),
        the number of samples and realizations, respectively.
    H : float
        Hurst exponent
    lam : float
        Lambda, intermittency parameter
    L : float
        Integral scale
    sigma : float
        Variance of process

    Returns
    -------
    mrw : ndarray
        Synthesized mrw realizations. If `shape` is scalar,
        fbm is ofshape (N,). Otherwise, it is of shape (N, R).

    References
    ----------
    .. [1] Bacry, Delour, Muzy, "Multifractal Random Walk", Physical Review E,
        2001
    '''

    try:
        N, R = shape
        do_squeeze = False
    except TypeError:  # shape is scalar
        N, R = shape, 1
        do_squeeze = True

    # Is 0.5 or 0 the lower bound ? Search biblio
    if not 0 <= H <= 1:
        raise ValueError('H must satisfy 0 <= H <= 1')

    if L > N:
        raise ValueError('Integral scale L is larger than data length N')

    # 1) Gaussian process w
    w = gaussian_w(N, R, L, lam, 1, method, z0[1])

    #   Adjust mean to ensure convergence of variance
    r = 1/2  # see Bacry, Delour & Muzy, Phys Rev E, 2001, page 4
    w = w - np.mean(w, axis=0) - r * lam**2 * np.log(L)

    # 2) fGn e
    e = fgn((N, R), H, sigma, method=method, z0=z0[0])

    # 3) mrw
    mrw = np.cumsum(e * np.exp(w), axis=0)

    return mrw.squeeze() if do_squeeze else mrw


def mrw_cumul(shape, c1, c2, L, **kwargs):
    '''
    Wrapper for mrw generation from cumulants.

    Parameters
    ----------
    shape : int | tuple(int)
        If scalar, it is the  number of samples. If tuple it is (N, R),
        the number of samples and realizations, respectively.
    c1 : float
        First order cumulant
    c2 : float
        Second order cumulant
    L : float
        Integral scale
    kwargs : dict
        Optional parameters passed to :obj:`mrw`

    Returns
    -------
    mrw : ndarray
        Synthesized mrw realizations. If `shape` is scalar,
        fbm is ofshape (N,). Otherwise, it is of shape (N, R).

    References
    ----------
    .. [1] Bacry, Delour, Muzy, "Multifractal Random Walk", Physical Review E,
        2001
    '''

    H = c1 + c2
    lam = np.sqrt(-c2)

    return mrw(shape, H, lam, L, **kwargs)


def skewed_mrw(shape, H, lam, L,  K0=1, alpha=1, sigma=1, dt=1, beta=1,
               do_mirror=False):
    '''
    Create skewed mrw as in Pochart & Bouchaud
    Assumes :math:`\\Delta_t=1`, so no parameter beta is needed.
    '''

    try:
        N, R = shape
        do_squeeze = False
    except TypeError:  # shape is scalar
        N, R = shape, 1
        do_squeeze = True

    # Is 0.5 or 0 the lower bound ? Search biblio
    if not 0 <= H <= 1:
        raise ValueError('H must satisfy 0 <= H <= 1')

    if L / dt > N:
        raise ValueError('Integral scale L/dt is larger than data length N')

    # 1) Gaussian process w
    w = gaussian_w(N, R, L, lam, dt)

    #   Adjust mean to ensure convergence of variance
    r = 1  # see Bacry, Delour & Muzy, Phys Rev E, 2001, page 4
    w = w - np.mean(w, axis=0) - r * lam**2 * np.log(L / dt)

    # 2) fGn e
    e = fgn((2*N + 1, R), H, sigma, dt)

    # 3) Correlate components
    past = skewness_convolution(e, K0, alpha, beta, dt)
    wtilde = w - past

    # 4) skewed mrw
    smrw = np.cumsum(e[N:] * np.exp(wtilde), axis=0)

    if do_squeeze:
        smrw = smrw.squeeze()

    if do_mirror:
        past_mirror = skewness_convolution(-e, K0, alpha, beta, dt)
        wtilde_mirror = w - past_mirror
        smrw_mirror = np.cumsum(-e[N:] * np.exp(wtilde_mirror), axis=0)
        if do_squeeze:
            smrw_mirror = smrw_mirror.squeeze()
        return smrw, smrw_mirror
    else:
        return smrw


def gaussian_w(N, R, L, lam, dt=1, method='cme', z0=None):
    '''
    Auxiliar function to create gaussian process w
    '''
    kmax = int(L / dt)
    k = np.arange(kmax)
    rho = np.ones((N))
    rho[:kmax] = L / (k + 1) / dt
    cov = (lam ** 2) * np.log(rho)
    if method == 'cme':
        w = gaussian_cme(cov, N, R, z0)
    elif method == 'chol':
        w = gaussian_chol(cov, N, R, z0)

    return w


def skewness_convolution(e, K0,  alpha, beta=1, dt=1):
    '''
    Noise e should be of length 2*N, with "N false past variables" at the
    beginning to avoid spurious correlations due to cutoffs in convolution.
    '''
    N, _ = e.shape
    N = N // 2

    tau = np.arange(1, N+1)
    Kbar = np.zeros((2*N))
    Kbar[1:N+1] = K0 / (tau**alpha) / (dt**beta)
    skew_conv = np.real(ifft(fft(Kbar[:, None], axis=0) *
                             fft(e, axis=0), axis=0))
    return skew_conv[N:]


def skewness_convolution_dumb(e, K0, alpha, beta=1, dt=1):
    '''
    Direct and inefficient calculation for testing purposes.
    Receives "true" input noise of size N.
    '''
    N, R = e.shape

    def K(i, j):
        return K0 / (j-i)**alpha / dt**beta

    scorr = np.zeros((N, R))
    for k in range(N):
        for i in range(k):
            scorr[k, :] += K(i, k) * e[i, :]
    return scorr


def mrw2D(shape, H, lam, L,  sigma=1):
    '''
    Create a realization of fractional Brownian motion using circulant
    matrix embedding.

    Parameters
    ----------
    shape : int | tuple(int)
        If scalar, it is the  number of samples. If tuple it is (N, R),
        the number of samples and realizations, respectively.
    H : float
        Hurst exponent
    lambda : float
        Intermittency parameter
    L : float
        Integral scale
    sigma : float
        Variance of process

    Returns
    -------
    mrw : ndarray
        Synthesized mrw realizations. If 'shape' is scalar,
        fbm is of shape (N,). Otherwise, it is of shape (N, N, R).

    References
    ----------
    .. [1] Bacry, Delour, Muzy, "Multifractal Random Walk", Physical Review E,
        2001
    '''

    try:
        N, R = shape
        # do_squeeze = False
    except TypeError:  # shape is scalar
        N, R = shape, 1
        # do_squeeze = True

    N = int(2 * np.ceil(N / 2))

    # dim = 2

    n = np.arange(-N // 2, N // 2)
    d = np.sqrt(n[:, None]**2 + n[None, :]**2)

    corr = lam**2 * np.log(np.maximum(L / (1 + d), 1))

    L = np.fft.fft2(corr)

    z1 = np.random.randn(N, N, R) + 1j * np.random.randn(N, N, R)
    w = np.exp(np.real(np.fft.ifft2(z1 * np.sqrt(L[..., None]), axes=(0, 1))))

    # Increment process:
    X = np.random.randn(N, N, R) * w

    # Fractional integration to produce motion:
    BX = fract_int_2d(X, H + 1)

    return BX, X


def fract_int_2d(x, alpha):
    '''
    Assumes size of x divisible by two
    '''
    N = x.shape[0]

    # Create Fourier filter
    k = np.arange(-N/2, N/2)

    d = np.sqrt(k[:, None]**2 + k[None, :]**2)
    mini = np.min(d[d != 0])
    d[d == 0] = mini
    filt = 1 / (d ** alpha)

    yhat = np.fft.fftshift(np.fft.fft2(x, axes=(0, 1)), axes=(0, 1))
    yhat *= filt[..., None]
    y = np.real(np.fft.ifft2(np.fft.ifftshift(yhat, axes=(0, 1)), axes=(0, 1)))
    return y
