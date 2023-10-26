"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

import warnings
from collections import namedtuple
from copy import deepcopy

import pywt
import numpy as np
from scipy.signal import convolve

from .structurefunction import StructureFunction
from .multiresquantity import MultiResolutionQuantity
from .utils import fast_power, get_filter_length, max_scale_bootstrap


def _check_formalism(p_exp):
    """
    Check formalism according to the value of p_exp
    """
    if p_exp is None:
        return 'wavelet coef'
    if np.isinf(p_exp):
        return 'wavelet leader'
    else:
        return 'wavelet p-leader'


def _estimate_eta_p(wt_coefs, p_exp, scaling_ranges, weighted):
    """
    Estimate the value of eta_p
    """

    wavelet_structure = StructureFunction(wt_coefs,
                                          np.array([p_exp]),
                                          scaling_ranges,
                                          weighted)

    # shape N_ranges, N_signals
    return wavelet_structure.zeta[0]


def decomposition_level_bootstrap(X, wt_name):
    """
    Determines maximum scale possible to perform bootstrapping

    Parameters
    ----------
    mrq: :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`

    """

    return max_scale_bootstrap(
        wavelet_analysis(X, wt_name=wt_name, p_exp=None)[0])


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

    Returns
    -------
    max_level : int
        The maximum scale
    """

    filter_len = get_filter_length(wt_name)

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

    if j2 is not None:
        if j2 > max_level and warn is not False:
            warnings.warn(
                "Value of j2 is higher than the maximum allowed level. "
                f"Max level and j2 set to {max_level}", UserWarning)

        max_level = min(j2, max_level)

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

    # high[np.isnan(high)] = np.inf
    # low[np.isnan(low)] = np.inf

    # index of first good value
    fp = len(high_filter) - 1
    # index of last good value
    lp = nj_temp  # len(high_filter)

    # replace border with nan
    high[0:fp] = np.nan
    high[lp:] = np.nan
    low[0:fp] = np.nan
    low[lp:] = np.nan

    # centering and subsampling
    # nwt = len(high_filter) // 2
    # nl = len(high_filter)
    # detail_idx = np.arange(1, nj_temp + 1, 2)
    # approx_idx = np.arange(1, nj_temp, 2) + 1

    # x0 = 2
    x0Appro = len(high_filter)  # 2*self.nb_vanishing_moments

    # centering and subsampling
    detail_idx = np.arange(0, nj_temp, 2) + 1
    approx_idx = np.arange(0, nj_temp, 2) + x0Appro - 1

    detail = high[detail_idx]
    approx = low[approx_idx]

    return detail, approx


def _find_sans_voisin(scale, detail, sans_voisin, formalism):

    if scale == 1:
        sans_voisin = detail

    else:

        max_index = int(np.floor(len(sans_voisin) / 2))
        detail = detail[:max_index]

        # print(detail[:2])

        sans_voisin = np.stack([detail,
                                sans_voisin[0:2*max_index:2],
                                sans_voisin[1:2*max_index:2]],
                               axis=0)

        # print(sans_voisin[:, :2])

        if formalism == 'wavelet p-leader':
            sans_voisin = np.sum(sans_voisin, axis=0)
        else:
            sans_voisin = np.max(sans_voisin, axis=0)

    return sans_voisin


def _compute_leaders(detail, sans_voisin, scale, formalism, p_exp,
                     eta_p_srange=None, eta_p_weighted=None, size=3):
    """
    Compute wavelet leaders
    """

    detail = np.abs(detail)

    if formalism == 'wavelet p-leader':
        detail = np.power(2., scale)*fast_power(detail, p_exp)

    sans_voisin = _find_sans_voisin(scale, detail, sans_voisin, formalism)

    # print(sans_voisin[:2], detail[:2])

    len_sv = len(sans_voisin)

    if size == 1:
        leaders = sans_voisin[None, :]

    elif size == 3:
        leaders = np.stack([sans_voisin[0:len_sv-2],
                            sans_voisin[1:len_sv-1],
                            sans_voisin[2:len_sv]],
                           axis=0)

    if formalism == 'wavelet p-leader':
        # import ipdb; ipdb.set_trace()
        leaders = np.sum(leaders, axis=0)
        leaders = fast_power(np.power(2., -scale)*leaders, 1/p_exp)
    else:
        leaders = np.max(leaders, axis=0)

    return leaders, sans_voisin


def compute_leaders(wt_coefs, gamint, p_exp, size=3):
    # TODO: call from wavelet_analysis

    formalism = _check_formalism(p_exp)

    sans_voisin = None
    wt_leaders = MultiResolutionQuantity(formalism, gamint, p_exp=p_exp,
                                         origin_mrq=wt_coefs)

    max_level = wt_coefs.j2_eff()

    for scale in range(1, max_level + 1):

        detail = wt_coefs.values[scale]

        leaders, sans_voisin = _compute_leaders(detail, sans_voisin,
                                                scale, formalism, p_exp,
                                                size=size)

        # remove infinite values and store wavelet leaders
        # finite_idx_wl = np.logical_not(np.isinf(np.abs(leaders)))
        finite_idx_wl = np.logical_not(np.isnan(np.abs(leaders)))
        # leaders[~finite_idx_wl] = np.nan

        if np.sum(finite_idx_wl, axis=0).min() < 3:
            max_level = scale-1
            break

        wt_leaders.add_values(leaders, scale)

    return wt_leaders


def compute_leaders2(wt_coefs, gamint, p_exp, size=3, idx_reject=None):

    formalism = _check_formalism(p_exp)

    wt_leaders = MultiResolutionQuantity(formalism, gamint,
                                         n_sig=wt_coefs.n_sig, p_exp=p_exp,
                                         origin_mrq=wt_coefs)

    max_level = wt_coefs.j2_eff()

    pleader_p = {}

    for scale in range(1, max_level + 1):

        # coefs = 2 ** scale * fast_power(np.abs(wt_coefs.values[scale]), p_exp)
        coefs = fast_power(np.abs(wt_coefs.values[scale]), p_exp)

        # if (idx_reject is not None and idx_reject[scale].sum() > 0
        #         and scale >= j1 and scale <= j2_reg):

        #     idx = idx_reject[scale]

        #     # coefs[idx] = np.nan

        #     # print(scale_contrib_reject_count)
        #     print(idx.sum())

        scale_contribution = np.stack([
            coefs[size-i:-(i-1) or None] for i in range(1, size+1)
        ], axis=0)

        if (idx_reject is not None
                and scale in idx_reject
                and idx_reject[scale].sum() > 0):

            idx = idx_reject[scale]
            scale_contribution[:, idx.squeeze().transpose()] = np.nan

        if scale == 1:

            leaders = np.sum(scale_contribution, axis=0)
            pleader_p[scale] = leaders
            # pleader_p[scale] = fast_power(np.power(2., -scale)*leaders, 1/p_exp)
            # print(pleader_p[scale].shape)
            continue

        # max_index = int(np.floor(len(coefs) / 2))

        # max_index = (pleader_p[scale-1].shape[0] - 3) // 2 * 2

        # print(pleader_p[scale-1][:-3:2].shape,
        #       pleader_p[scale-1][3::2].shape)

        lower_contribution = np.stack([
            pleader_p[scale-1][:-size:2],
            pleader_p[scale-1][size::2]
        ], axis=0)

        # assert scale_contribution.shape[1] == lower_contribution.shape[1],\
        #     print(scale_contribution.shape, lower_contribution.shape, scale)
        #     print(pleader_p[scale-1].shape, coefs.shape, max_index)

        # print(max_index, coefs.shape[0], pleader_p[scale-1].shape[0],
        #       scale_contribution[:, :max_index // 2].shape)

        max_index = lower_contribution.shape[1]

        # print(scale_contribution.shape, lower_contribution.shape)

        leaders = np.sum(np.r_[
            scale_contribution[:, :max_index],
            .5 * lower_contribution
        ], axis=0)
        pleader_p[scale] = leaders

        # finite_idx_wl = np.logical_not(np.isnan(np.abs(leaders)))
        # leaders[~finite_idx_wl] = np.nan

        # if np.sum(finite_idx_wl, axis=0).min() < 3:
        #     max_level = scale-1
        #     break

    for scale in range(1, max_level + 1):

        leaders = fast_power(pleader_p[scale], 1/p_exp)
        wt_leaders.add_values(leaders, scale)

    if formalism == 'wavelet p-leader':

        working_coefs = deepcopy(wt_coefs)

        if idx_reject is not None:
            for scale in idx_reject:
                working_coefs.values[scale][1:-1][idx_reject[scale][0].squeeze().transpose()] = np.nan

    return wt_leaders


WaveletTransform = namedtuple('WaveletTransform', ['wt_coefs',
                                                   'wt_leaders',
                                                   'j2_eff'])
r"""Aggregates the output of wavelet analysis

Attributes
----------
wt_coefs : :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`
    Wavelet coefficients
wt_leaders : :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`
    Wavelet leaders, or p-leaders, depending on the value of ``p_exp`` passed
j2_eff : int
    Maximum scale effectively used during the computation of the coefficients
eta_p : float
    Estimated value of :math:`\eta_p`, before applying p-leader correction
"""


def _wavelet_coef_analysis(approx, max_level, high_filter, low_filter,
                           normalization, gamint, j2, wt_name):

    wt_coefs = MultiResolutionQuantity('wavelet coef', gamint, wt_name)
    wt_leaders = None

    for scale in range(1, max_level + 1):

        detail, approx = filtering(approx, high_filter, low_filter)

        # normalization
        detail = detail*2**(scale*(0.5-1/normalization))

        # fractional integration
        detail = detail*2.0**(gamint*scale)

        # remove infinite values and store wavelet coefficients
        # finite_idx_coef = np.logical_not(np.isinf(np.abs(detail)))
        finite_idx_coef = np.logical_not(np.isnan(np.abs(detail)))

        # print(np.sum(finite_idx_coef, axis=0).min())

        if np.sum(finite_idx_coef, axis=0).min() < 3:
            max_level = scale-1
            break

        # if 0 in np.sum(finite_idx_coef, axis=0):

        # detail[~finite_idx_coef] = np.nan
        wt_coefs.add_values(detail, scale)

    # "effective" j2, used in linear regression
    j2_eff = int(min(max_level, j2) if j2 is not None else max_level)

    return WaveletTransform(wt_leaders=wt_leaders,
                            wt_coefs=wt_coefs,
                            j2_eff=j2_eff)


def wavelet_analysis(signal, p_exp=None, wt_name='db3', j2=None,
                     gamint=0.0, normalization=1):
    """
    Compute wavelet coefficient and wavelet leaders.


    Parameters
    ----------
    signal : ndarray, shape (n_samples,) | (n_samples, n_realisations)
        Time series to analyze.

    p_exp : float | np.inf | None
        Determines the formalism to be used: None means only wavelet coefs
        will be computed, np.inf means wavelet leaders will also be computed,
        and an int sets the value of the p exponent implying a wavelet p-leader
        formalism.

    wt_name : str
        Name of the wavelet function to use, as defined in the pywavelet
        package [1]_. The default value of ``'db3'`` means Daubechies with 3
        vanishing moments.

    j1 : int
        Lower bound of the scale range on which to estimate :math:`\\eta_p` in
        p-leader correction.

    j2 : int | None
        Upper bound of the scale range for which wavelet coefficients
        will be computed. If None, it will automatically be set to the
        highest value possible.

    gamint : float
        Fractional integration coefficient :math:`\\gamma_{\\textrm{int}}`

    normalization : int
        Norm to use on the wavelet coefficients, see notes for more details.

    weighted : str | None
        Whether to perform weighted linear regression, used only when
        computing p-leaders for when estimating :math:`\\eta_p` in p-leader
        correction

    j2_reg: int
        Upper bound of the scale range on which to estimate :math:`\\eta_p' in p-leader correction

    Returns
    -------
    WaveletTransform
        Namedtuple containing the computed wavelet coefs, potential wavelet
        leaders, and the effective maximum scale used

    Notes
    -----
    When computing the wavelet coefficients, the values corrupted
    by border effects are set to infinity (np.inf).

    This makes it easier to compute the wavelet leaders, since
    corrupted values will also be infinite and can be removed.

    .. note:: Wavelet coefficients are usually L^1 normalized [2]_, which is
              achieved by setting ``normalization=1``.

    References
    ----------

    .. [1] https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html

    .. [2] \
        http://perso-math.univ-mlv.fr/users/jaffard.stephane/pdf/Mandelbrot.pdf
        , page5
    """

    if len(signal.shape) == 1:
        signal = signal[:, None]

    # Initialize the filter
    wavelet = pywt.Wavelet(wt_name)
    # Investigate why -1
    high_filter = -1*np.array(wavelet.dec_hi)[:, None]
    low_filter = np.array(wavelet.dec_lo)[:, None]

    formalism = _check_formalism(p_exp)

    max_level = _decomposition_level(signal, len(high_filter), j2)
    approx = signal

    if formalism == 'wavelet coef':
        return _wavelet_coef_analysis(approx, max_level, high_filter,
                                      low_filter, normalization, gamint, j2,
                                      wt_name)

    # Initialize structures
    wt_coefs = MultiResolutionQuantity('wavelet coef', gamint, wt_name,
                                       n_sig=signal.shape[1])
    wt_leaders = MultiResolutionQuantity(formalism, gamint, wt_name,
                                         n_sig=signal.shape[1], p_exp=p_exp,
                                         origin_mrq=wt_coefs)

    sans_voisin = None

    for scale in range(1, max_level + 1):

        detail, approx = filtering(approx, high_filter, low_filter)

        # normalization
        detail = detail*2**(scale*(0.5-1/normalization))

        # fractional integration
        detail = detail*2.0**(gamint*scale)

        # finite_idx_coef = np.logical_not(np.isinf(np.abs(detail)))
        finite_idx_coef = np.logical_not(np.isnan(np.abs(detail)))

        if np.sum(finite_idx_coef, axis=0).min() < 3:
            max_level = scale-1
            break

        # if 0 in np.sum(finite_idx_coef, axis=0):
        #     max_level = scale-1
        #     break

        leaders, sans_voisin = _compute_leaders(detail, sans_voisin,
                                                scale, formalism, p_exp)

        # remove infinite values and store wavelet coefficients
        # detail[~finite_idx_coef] = np.nan
        wt_coefs.add_values(detail, scale)

        # remove infinite values and store wavelet leaders
        # finite_idx_wl = np.logical_not(np.isinf(np.abs(leaders)))
        finite_idx_wl = np.logical_not(np.isnan(np.abs(leaders)))
        # leaders[~finite_idx_wl] = np.nan

        if np.sum(finite_idx_wl, axis=0).min() < 3:
            max_level = scale-1
            break

        wt_leaders.add_values(leaders, scale)

    if max_level == 0:
        return WaveletTransform(None, None, max_level)

    return WaveletTransform(wt_leaders=wt_leaders,
                            wt_coefs=wt_coefs,
                            j2_eff=max_level)


def compute_wse(wt_coefs, theta=0.5):

    wse_coef = MultiResolutionQuantity('weak scaling exponent', wt_coefs.gamint, wt_coefs.wt_name)

    for scale, dwt in wt_coefs.values.items():
        lower_scale = max(1,int(scale-scale**(theta))) # On définit une tranche d'échelle J/sqrt(J)
        # lower_scale = np.min(lower_scale,scale)  # On prend en compte où on dépasse le nombre d'échelle max
        # J2 = 1  # Dans les cas des leaders J2=1

        wse = np.zeros_like(dwt)  # On initialise le max à 0
        for k in range(dwt.shape[0]): # On calcule les coefficient l(J1,k)

            for j in range(scale,lower_scale-1,-1): # On parcourt les tranches d'échelle allant de J2 à J1
                packet_size = (scale-j)+1  # On prend des paquets de coefficients qui varie
                # packet_size = 1  # Pour calculer les leaders

                cwav = wt_coefs.values[j] # On stock tous les coefs en ondelettes
                nwav = cwav.shape[0]  # On compte le nombre de coefs

                left_bound = int(max(1, 2^(scale-j) * (k-packet_size)))-1 # On calcule la borne de gauche
                right_bound = int(min(nwav,2^(scale-j) * (k+packet_size)))-1 # On calcule la borne de droite

                if right_bound <= left_bound:
                    continue

                wse[k] = np.nanmax(np.r_[wse[k], np.nanmax(abs(cwav[left_bound:right_bound]), axis=0)], axis=0) # On détermine le WSE

        wse_coef.add_values(wse, scale)

    return wse_coef
