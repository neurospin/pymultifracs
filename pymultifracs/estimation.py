"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.linear_model import LinearRegression

from .regression import linear_regression, prepare_regression, prepare_weights
from . import scalingfunction
from . import utils


def estimate_hmin(mrq, scaling_ranges, weighted, idx_reject, warn=True,
                  return_y=False):
    """
    Estimate the value of the uniform regularity exponent hmin using
    wavelet coefficients.
    """

    x, n_ranges, j_min, j_max, *_ = prepare_regression(
        scaling_ranges, np.array([*mrq.values])
    )

    if weighted == 'bootstrap' and mrq.bootstrapped_obj is not None:

        std = np.std(
            mrq.bootstrapped_obj.sup_coeffs(
                n_ranges, j_max, j_min, scaling_ranges, idx_reject
                ).reshape(j_max-j_min+1, len(scaling_ranges), mrq.n_sig, -1),
            axis=-1)[None, :]

    else:
        std = None

    sup_coeffs = mrq.sup_coeffs(
        n_ranges, j_max, j_min, scaling_ranges, idx_reject)

    y = np.log2(sup_coeffs)[None, :]
    
    w = prepare_weights(mrq, weighted, n_ranges, j_min, j_max,
                        scaling_ranges, y, std=std)

    slope, intercept = linear_regression(x, y, w)

    hmin = slope[0]

    # warning
    if 0 in hmin and warn:
        warnings.warn(f"h_min = {hmin} < 0. gamint should be increased")

    if return_y:
        return hmin, intercept[0], y[0]

    return hmin, intercept[0]


def _estimate_eta_p(wt_coefs, p_exp, scaling_ranges, weighted, idx_reject):
    """
    Estimate the value of eta_p
    """

    bootstrapped_obj = None

    if weighted == 'bootstrap':
        ws_boot = scalingfunction.StructureFunction(
            mrq=wt_coefs.bootstrapped_obj, q=np.array([p_exp]),
            scaling_ranges=scaling_ranges, weighted=None)
        bootstrapped_obj = utils.MFractalVar(ws_boot, None, None)

    wavelet_structure = scalingfunction.StructureFunction(
        mrq=wt_coefs, q=np.array([p_exp]), scaling_ranges=scaling_ranges,
        weighted=weighted, idx_reject=idx_reject,
        bootstrapped_obj=bootstrapped_obj)

    # shape N_ranges, N_signals
    return wavelet_structure.zeta[0]


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
    reg_y = map(lambda x: hmin*x + intercept, reg_x)

    legend = f'$h_\\mathrm{min}$ = {hmin:.5f}'
    plt.plot(reg_x, reg_y, color='k', linestyle='-', linewidth=2, label=legend)
    plt.legend()
    plt.draw()

    plt.show()


# def compute_hurst(wt_coefs, j1, j2, weighted):
#     """
#     Estimate the Hurst exponent using the wavelet structure function for q=2
#     """

#     structure_dwt = scalingfunction.StructureFunction(wt_coefs,
#                                       np.array([2.0]),
#                                       j1,
#                                       j2,
#                                       weighted)

#     log2_Sj_2 = np.log2(structure_dwt.values[0, :])  # log2(S(j, 2))
#     hurst_structure = log2_Sj_2
#     hurst = structure_dwt.zeta[0]/2

#     return hurst, structure_dwt, hurst_structure
