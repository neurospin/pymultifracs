import warnings
from collections import namedtuple

import pywt
import numpy as np
from scipy.signal import convolve

from .structurefunction import StructureFunction
from .multiresquantity import MultiResolutionQuantity
from .utils import smart_power


def _check_formalism(p_exp):
    """
    Check formalism according to the value of p_exp
    """
    if p_exp is None:
        return 'wcmf'
    if np.isinf(p_exp):
        return 'wlmf'
    else:
        return 'p-leader'


def _estimate_eta_p(wt_coefs, p_exp, j1, j2_eff, weighted):
    """
    Estimate the value of eta_p
    """
    wavelet_structure = StructureFunction(wt_coefs,
                                          np.array([p_exp]),
                                          j1, j2_eff,
                                          weighted,
                                          stat_fun='mean')

    return wavelet_structure.zeta[0]


def _correct_leaders(wt_coefs, wt_leaders, p_exp, j1, j2_eff,
                     weighted, max_level):
    """
    Correct p-leaders for nonlinearity (according to the Matlab toolbox)
    """

    eta_p = _estimate_eta_p(wt_coefs, p_exp, j1, j2_eff, weighted)

    if eta_p <= 0:
        warnings.warn(f"eta(p) = {eta_p} <= 0, p-Leaders correction was not\
                        applied. A smaller value of p (or larger value of\
                        gamint) should be selected.")

    else:

        JJ = np.arange(1, max_level + 1)
        J1LF = 1
        JJ0 = JJ - J1LF + 1
        zqhqcorr = np.log2((1 - np.power(2., -JJ0 * eta_p))
                           / (1 - np.power(2., -eta_p)))
        ZPJCorr = np.power(2, (-1.0 / p_exp) * zqhqcorr)

        for ind_j, j in enumerate(JJ):
            wt_leaders.values[j] = \
                wt_leaders.values[j]*ZPJCorr[ind_j]

    return wt_leaders


def decomposition_level(length, wt_name):
    """
    Checks the maximum scale which can be used to decompose a signal
    of given length

    Parameters
    ----------

    length: int
        Length of the signal considered

    wt_name: str
        Name of the wavelet function to use, following the pywavelet convention
    """

    wt = pywt.Wavelet(wt_name)
    filter_len = len(wt.dec_hi)

    max_level = int(np.floor(np.log2(length / (filter_len + 1))))
    max_level = min(int(np.floor(np.log2(length))), max_level)

    return max_level


def _decomposition_level(signal, filter_len, j2, warn=True):
    """
    Check maximum decomposition level
    """

    length = len(signal)
    max_level = int(np.floor(np.log2(length / (filter_len + 1))))
    max_level = min(int(np.floor(np.log2(length))), max_level)

    # Warning if j2 is greater than max_level
    if j2 is not None and j2 > max_level and warn is not False:
        warnings.warn("Value of j2 is higher than the maximum allowed level. "
                      f"Max level and j2 set to {max_level}", UserWarning)

    # max_level = min(max_level, self.j2)

    return max_level


def filtering(approx, high_filter, low_filter):
    """
    """

    nj_temp = len(approx)

    # apply filters
    # note: 'direct' method MUST be used, since there are elements
    # that are np.inf inside `approx`
    high = convolve(approx, high_filter, mode='full', method='direct')
    low = convolve(approx, low_filter, mode='full', method='direct')

    high[np.isnan(high)] = np.inf
    low[np.isnan(low)] = np.inf

    # index of first good value
    fp = len(high_filter) - 2
    # index of last good value
    lp = nj_temp - 1

    # replace border with Inf
    high[0:fp] = np.inf
    high[lp+1:] = np.inf
    low[0:fp] = np.inf
    low[lp+1:] = np.inf

    # centering and subsampling
    detail_idx = np.arange(1, nj_temp + 1, 2)
    approx_idx = np.arange(-1, nj_temp - 1, 2) + len(high_filter)

    detail = high[detail_idx]
    approx = low[approx_idx]

    return detail, approx


def _find_sans_voisin(scale, detail, sans_voisin, formalism):

    if scale == 1:
        sans_voisin = detail

    else:

        max_index = int(np.floor(len(sans_voisin) / 2))
        detail = detail[:max_index]

        sans_voisin = np.vstack((detail,
                                 sans_voisin[0:2*max_index:2],
                                 sans_voisin[1:2*max_index:2]))

        if formalism == 'p-leader':
            sans_voisin = sans_voisin.sum(axis=0)
        else:
            sans_voisin = sans_voisin.max(axis=0)

    return sans_voisin


def _compute_leaders(detail, sans_voisin, scale, formalism, p_exp):
    """
    Compute wavelet leaders
    """

    detail = np.abs(detail)

    if formalism == 'p-leader':
        detail = np.power(2., scale)*smart_power(detail, p_exp)

    sans_voisin = _find_sans_voisin(scale, detail, sans_voisin, formalism)

    len_sv = len(sans_voisin)
    leaders = np.vstack((sans_voisin[0:len_sv-2],
                         sans_voisin[1:len_sv-1],
                         sans_voisin[2:len_sv]))

    if formalism == 'p-leader':
        leaders = leaders.sum(axis=0)
        leaders = smart_power(np.power(2., -scale)*leaders, 1/p_exp)
    else:
        leaders = leaders.max(axis=0)

    return leaders, sans_voisin


WaveletTransform = namedtuple('WaveletTransform', ['wt_coefs',
                                                   'wt_leaders',
                                                   'j2_eff'])


def wavelet_analysis(signal, p_exp=None, wt_name='db3', j1=1, j2=10,
                     gamint=0.0, normalization=1, weighted=True):
    """
    Compute all the wavelet coefficients from scales 1 to self.j2

    Wavelet coefficients are usually L^1 normalized, see page 5:
        http://perso-math.univ-mlv.fr/users/jaffard.stephane/pdf/Mandelbrot.pdf

    Explanation:
        When computing the wavelet coefficients, the values corrupted
        by border effects are set to infinity (np.inf).

        This makes it easier to compute the wavelet leaders, since
        corrupted values will also be infinite and can be removed.

    Parameters
    ----------
    signal: 1D-array_like
        Time series of signal to analyze

    p_exp: int | np.inf | default None
        Determines the formalism to be used:
            -None means wavelet coefficient
            -np.inf means wavelet leader
            -int value means p-leader

    wt_name: str 
        name of the wavelet function to use, as defined in the pywavelet
        package

    j1: int
        lower bound of the scale range on which to estimate eta_p in
        p-leader correction

    j2: int
        upper bound of the scale range for which wavelet coefficients
        will be conputed

    gamint: float
        fractional integration coefficient

    normalization: int
        norm to use

    weighted: bool
    """

    # Initialize the filter
    wavelet = pywt.Wavelet(wt_name)
    high_filter = -1*np.array(wavelet.dec_hi)
    low_filter = np.array(wavelet.dec_lo)

    formalism = _check_formalism(p_exp)

    max_level = _decomposition_level(signal, len(high_filter), j2)

    # Initialize structures 1
    wt_coefs = MultiResolutionQuantity(formalism)
    wt_leaders = MultiResolutionQuantity(formalism)

    sans_voisin = None

    approx = signal

    for scale in range(1, max_level + 1):

        detail, approx = filtering(approx, high_filter, low_filter)

        # normalization
        detail_scale = detail*2**(scale*(0.5-1/normalization))

        # fractional integration
        detail_scale = detail_scale*2.0**(gamint*scale)

        # remove infinite values and store wavelet coefficients
        finite_idx_coef = np.logical_not(np.isinf(np.abs(detail_scale)))

        if np.sum(finite_idx_coef) == 0:
            max_level = scale-1
            break
        wt_coefs.add_values(detail_scale[finite_idx_coef], scale)

        # Compute wavelet leaders if needed
        if formalism in ['wlmf', 'p-leader']:

            leaders, sans_voisin = _compute_leaders(detail_scale, sans_voisin,
                                                    scale, formalism, p_exp)

            # remove infinite values and store wavelet leaders
            finite_idx_wl = np.logical_not(np.isinf(np.abs(leaders)))

            if np.sum(finite_idx_wl) == 0:
                max_level = scale-1
                break
            wt_leaders.add_values(leaders[finite_idx_wl], scale)

    # "effective" j2, used in linear regression
    j2_eff = int(min(max_level, j2) if j2 is not None else max_level)

    if formalism == 'p-leader':
        wt_leaders = _correct_leaders(wt_coefs, wt_leaders, p_exp, j1, j2_eff,
                                      weighted, max_level)

    return WaveletTransform(wt_leaders=wt_leaders,
                            wt_coefs=wt_coefs,
                            j2_eff=j2_eff)
