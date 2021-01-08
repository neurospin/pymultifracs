import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from .utils import linear_regression
from .structurefunction import StructureFunction


def _estimate_hmin_classic(wavelet_coeffs, j1, j2_eff, weighted, warn=True):
    """
    Estimate the value of the uniform regularity exponent hmin using
    wavelet coefficients.
    """
    sup_coeffs = np.zeros(j2_eff - j1 + 1)

    for j in range(j1, j2_eff+1):
        c_j = np.abs(wavelet_coeffs.values[j])
        sup_c_j = c_j.max()
        sup_coeffs[j-j1] = sup_c_j

    log_sup_coeffs = np.log2(sup_coeffs)

    # x, y and weights for linear regression
    x = np.arange(j1, j2_eff+1)
    y = log_sup_coeffs
    if weighted:
        nj = wavelet_coeffs.get_nj_interv(j1, j2_eff)
    else:
        nj = np.ones(len(x))

    # linear regression
    slope, intercept = linear_regression(x, y, nj)
    hmin = slope

    # warning
    if hmin < 0 and warn:
        warnings.warn(f"h_min = {hmin} < 0. gamint should be increased")

    return slope, intercept


def estimate_hmin(wt_coefs, j1, j2_eff, weighted, warn=True,
                  return_y=False):
    """
    Estimate the value of the uniform regularity exponent hmin using
    wavelet coefficients.
    """
    sup_coeffs = np.zeros(j2_eff - j1 + 1)

    for j in range(j1, j2_eff+1):
        c_j = np.abs(wt_coefs.values[j])
        sup_c_j = c_j.max()
        sup_coeffs[j-j1] = sup_c_j

    # x, y and weights for linear regression
    x = np.arange(j1, j2_eff+1)
    y = np.log2(sup_coeffs)
    if weighted:
        nj = wt_coefs.get_nj_interv(j1, j2_eff)
    else:
        nj = None

    # linear regression
    x = x.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(x, y, nj)
    hmin = regressor.coef_[0]

    # warning
    if hmin < 0 and warn:
        warnings.warn(f"h_min = {hmin} < 0. gamint should be increased")

    if return_y:
        return hmin, regressor.intercept_, y

    return hmin, regressor.intercept_


def plot_hmin(wt_coefs, j1, j2_eff, weighted, warn=True):

    hmin, intercept, y = estimate_hmin(wt_coefs, j1, j2_eff, weighted, warn)
    x = np.arange(j1, j2_eff+1)

    # plot log_sup_coeffs
    plt.figure('hmin')

    plt.plot(x, y, 'r--.')
    plt.xlabel('j')
    plt.ylabel(r'$\log_2(\sup_k |d(j,k)|)$')
    plt.suptitle(r'$h_\mathrm{min}$')

    plt.draw()
    plt.grid()

    # plot regression line
    reg_x = [j1, j2_eff]
    f = lambda x: hmin*x + intercept
    reg_y = map(f, reg_x)

    legend = f'$h_\\mathrm{min}$ = {hmin:.5f}'
    plt.plot(reg_x, reg_y, color='k', linestyle='-', linewidth=2, label=legend)
    plt.legend()
    plt.draw()

    plt.show()


def compute_hurst(wt_coefs, j1, j2, weighted):
    """
    Estimate the Hurst exponent using the wavelet structure function for q=2
    """

    structure_dwt = StructureFunction(wt_coefs,
                                      np.array([2.0]),
                                      j1,
                                      j2,
                                      weighted)

    log2_Sj_2 = np.log2(structure_dwt.values[0, :])  # log2(S(j, 2))
    hurst_structure = log2_Sj_2
    hurst = structure_dwt.zeta[0]/2

    return hurst, structure_dwt, hurst_structure
