"""
Authors: Merlin Dumeur <merlin@dumeur.net>
         Omar D. Domingues <omar.darwiche-domingues@inria.fr>
"""

import warnings
from math import ceil, floor

import pywt
import numpy as np

from . import multiresquantity
from .utils import fast_power, max_scale_bootstrap


# def _check_formalism(p_exp):
#     """
#     Check formalism according to the value of p_exp
#     """
#     if p_exp is None:
#         return 'wavelet coef'
#     if np.isinf(p_exp):
#         return 'wavelet leader'
#     else:
#         return 'wavelet p-leader'

def decomposition_level_bootstrap(X, wt_name):
    """
    Determines maximum scale possible to perform bootstrapping

    Parameters
    ----------
    mrq: :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`

    """

    return max_scale_bootstrap(wavelet_analysis(X, wt_name=wt_name))


# def decomposition_level(length, wt_name):
#     """
#     Checks the maximum scale which can be used to decompose a signal
#     of given length

#     Parameters
#     ----------
#     length: int
#         Length of the signal considered
#     wt_name: str
#         Name of the wavelet function to use, following the pywavelet
#         convention

#     Returns
#     -------
#     max_level : int
#         The maximum scale
#     """

#     filter_len = get_filter_length(wt_name)

#     max_level = int(np.floor(np.log2(length / (filter_len + 1))))
#     max_level = min(int(np.floor(np.log2(length))), max_level)

#     return max_level


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


# def filtering(approx, high_filter, low_filter):
#     """
#     """

#     nj_temp = len(approx)

#     # apply filters
#     # note: 'direct' method MUST be used, since there are elements
#     # that are np.inf inside `approx`
#     high = signal.convolve(approx, high_filter, mode='full', method='direct')
#     low = signal.convolve(approx, low_filter, mode='full', method='direct')

#     # high[np.isnan(high)] = np.inf
#     # low[np.isnan(low)] = np.inf

#     # index of first good value
#     fp = len(high_filter) - 1
#     # index of last good value
#     lp = nj_temp  # len(high_filter)

#     # replace border with nan
#     high[0:fp] = np.nan
#     high[lp:] = np.nan
#     low[0:fp] = np.nan
#     low[lp:] = np.nan

#     # centering and subsampling
#     # nwt = len(high_filter) // 2
#     # nl = len(high_filter)
#     # detail_idx = np.arange(1, nj_temp + 1, 2)
#     # approx_idx = np.arange(1, nj_temp, 2) + 1

#     # x0 = 2
#     x0Appro = len(high_filter)  # 2*self.nb_vanishing_moments

#     # centering and subsampling
#     detail_idx = np.arange(0, nj_temp, 2) + 1
#     approx_idx = np.arange(0, nj_temp, 2) + x0Appro - 1

#     detail = high[detail_idx]
#     approx = low[approx_idx]

#     # detail = detail[fp:-fp]

#     return detail, approx


def _filtering2(approx, wt):

    # mode = 'per' if standard == 'matlab' else 'zero'
    mode = 'zero'
    low, high = pywt.dwt(approx, wt, mode=mode, axis=0)

    # if mode == 'per':
    #     # index of first good value
    #     fp = ceil(wt.dec_len / 2) - 1
    #     lp = - fp
    # if mode == 'zero':
    fp = ceil((wt.dec_len - 1) / 2) - 1
    # index of last good value
    lp = -fp

    # replace border with nan
    high[:fp] = np.nan
    high[lp:] = np.nan
    low[:fp] = np.nan
    low[lp:] = np.nan

    if approx.shape[0] % 2 == 1:
        return -high[:-2], low[fp:lp]

    if lp == -1:
        low_slice = np.s_[fp:]
    else:
        low_slice = np.s_[fp:lp+1]

    return -high[:-1], low[low_slice]


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


# def _compute_leaders(detail, sans_voisin, scale, formalism, p_exp,
#                      eta_p_srange=None, eta_p_weighted=None, size=3):
#     """
#     Compute wavelet leaders
#     """

#     detail = np.abs(detail)

#     if formalism == 'wavelet p-leader':
#         detail = np.power(2., scale)*fast_power(detail, p_exp)

#     sans_voisin = _find_sans_voisin(scale, detail, sans_voisin, formalism)

#     # print(sans_voisin[:2], detail[:2])

#     len_sv = len(sans_voisin)

#     if size == 1:
#         leaders = sans_voisin[None, :]

#     elif size == 3:
#         leaders = np.stack([sans_voisin[0:len_sv-2],
#                             sans_voisin[1:len_sv-1],
#                             sans_voisin[2:len_sv]],
#                            axis=0)

#     if formalism == 'wavelet p-leader':
#         # import ipdb; ipdb.set_trace()
#         leaders = np.sum(leaders, axis=0)
#         leaders = fast_power(np.power(2., -scale)*leaders, 1/p_exp)
#     else:
#         leaders = np.max(leaders, axis=0)

#     return leaders, sans_voisin


# def compute_leaders2(wt_coefs, gamint, p_exp, size=3):

#     formalism = _check_formalism(p_exp)

#     sans_voisin = None
#     wt_leaders = multiresquantity.WaveletLeader(
#         gamint=gamint, p_exp=p_exp, origin_mrq=wt_coefs, interval_size=size,
#         wt_name=wt_coefs.wt_name)

#     max_level = wt_coefs.j2_eff()

#     for scale in range(1, max_level + 1):

#         detail = wt_coefs.values[scale]

#         leaders, sans_voisin = _compute_leaders(detail, sans_voisin,
#                                                 scale, formalism, p_exp,
#                                                 size=size)

#         # remove infinite values and store wavelet leaders
#         # finite_idx_wl = np.logical_not(np.isinf(np.abs(leaders)))
#         finite_idx_wl = np.logical_not(np.isnan(np.abs(leaders)))
#         # leaders[~finite_idx_wl] = np.nan

#         if np.sum(finite_idx_wl, axis=0).min() < 3:
#             max_level = scale-1
#             break

#         wt_leaders._add_values(leaders, scale)

#     return wt_leaders


def compute_leaders(wt_coefs, p_exp=np.inf, size=3):
    """
    Computes the wavelet (p)-leaders from the wavelet coefficients
    """

    if size % 2 == 0:
        raise ValueError(
            f'Interval size needs to be an odd integer, currently is {size}')

    if p_exp is None:
        return wt_coefs

    wt_leaders = multiresquantity.WaveletLeader(
        gamint=wt_coefs.gamint, n_channel=wt_coefs.n_channel, p_exp=p_exp,
        origin_mrq=wt_coefs, interval_size=size, wt_name=wt_coefs.wt_name)

    max_level = wt_coefs.j2_eff()

    leader_flag = p_exp == np.inf

    if leader_flag:
        p_exp = 1

    pleader_p = {j: np.zeros_like(wt_coefs.values[j]) for j in wt_coefs.values}

    for scale in range(1, max_level + 1):

        # coefs = 2 ** scale * fast_power(np.abs(wt_coefs.values[scale]),
        #                                 p_exp)
        coefs = fast_power(
            np.abs(wt_coefs.values[scale]), p_exp)

        # if (idx_reject is not None and idx_reject[scale].sum() > 0
        #         and scale >= j1 and scale <= j2_reg):

        #     idx = idx_reject[scale]

        #     # coefs[idx] = np.nan

        #     # print(scale_contrib_reject_count)
        #     print(idx.sum())

        scale_contribution = np.zeros((size, *coefs.shape)) + np.nan

        if size > 1:
            idx_size = np.s_[(size-1)//2:-((size-1)//2)]
        else:
            idx_size = np.s_[:]

        scale_contribution[:, idx_size] = np.stack([
            coefs[size-i:-(i-1) or None] for i in range(1, size+1)
        ], axis=0)

        # if (idx_reject is not None
        #         and scale in idx_reject
        #         and idx_reject[scale].sum() > 0):

        #     idx = idx_reject[scale]
        #     scale_contribution[:, idx.squeeze().transpose()] = np.nan

        if scale == 1:

            leaders = np.sum(scale_contribution, axis=0)
            pleader_p[scale] = leaders

            # pleader_p[scale] = fast_power(np.power(2., -scale)*leaders,
            #                               1/p_exp)
            # print(pleader_p[scale].shape)
            continue

        max_index = pleader_p[scale-1].shape[0] // 2

        # max_index = (pleader_p[scale-1].shape[0] - 3) // 2 * 2

        # print(pleader_p[scale-1][:-3:2].shape,
        #       pleader_p[scale-1][3::2].shape)

        lower_contribution = np.zeros((2, *coefs.shape)) + np.nan

        lower_contribution[0][:max_index] = \
            pleader_p[scale-1][::2][:max_index]
        lower_contribution[1][:max_index] = \
            pleader_p[scale-1][1::2][:max_index]

        # lower_contribution = np.stack([
        #     pleader_p[scale-1][:-size:2],
        #     pleader_p[scale-1][size::2]
        # ], axis=0)

        # assert scale_contribution.shape[1] == lower_contribution.shape[1],\
        #     print(scale_contribution.shape, lower_contribution.shape, scale)
        #     print(pleader_p[scale-1].shape, coefs.shape, max_index)

        # print(max_index, coefs.shape[0], pleader_p[scale-1].shape[0],
        #       scale_contribution[:, :max_index // 2].shape)

        # max_index = lower_contribution.shape[1]

        # print(scale_contribution.shape, lower_contribution.shape)

        if leader_flag:

            pleader_p[scale] = np.max(np.r_[
                scale_contribution,
                .5 * lower_contribution
            ], axis=0)

        else:

            leaders = np.sum(np.r_[
                scale_contribution,
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

        # mask_nan = ((~np.isnan(leaders)).sum(axis=0) >= 3) + np.nan
        # leaders *= mask_nan[None, :]

        wt_leaders._add_values(leaders, scale)

    return wt_leaders


# WaveletTransform = namedtuple('WaveletTransform', ['wt_coefs',
#                                                    'wt_leaders',
#                                                    'j2_eff'])
# r"""Aggregates the output of wavelet analysis

# Attributes
# ----------
# wt_coefs : :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`
#     Wavelet coefficients
# wt_leaders : :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`
#     Wavelet leaders, or p-leaders, depending on the value of ``p_exp`` passed
# j2_eff : int
#     Maximum scale effectively used during the computation of the coefficients
# eta_p : float
#     Estimated value of :math:`\eta_p`, before applying p-leader correction
# """


# def _wavelet_coef_analysis(approx, max_level, high_filter, low_filter,
#                            normalization, gamint, j2, wt_name):

#     wt_coefs = MultiResolutionQuantity('wavelet coef', gamint, wt_name)
#     wt_leaders = None

#     for scale in range(1, max_level + 1):

#         detail, approx = filtering(approx, high_filter, low_filter)

#         # normalization
#         detail = detail*2**(scale*(0.5-1/normalization))

#         # fractional integration
#         detail = detail*2.0**(gamint*scale)

#         # remove infinite values and store wavelet coefficients
#         # finite_idx_coef = np.logical_not(np.isinf(np.abs(detail)))
#         finite_idx_coef = np.logical_not(np.isnan(np.abs(detail)))

#         # print(np.sum(finite_idx_coef, axis=0).min())

#         if np.sum(finite_idx_coef, axis=0).min() < 3:
#             max_level = scale-1
#             break

#         # if 0 in np.sum(finite_idx_coef, axis=0):

#         # detail[~finite_idx_coef] = np.nan
#         wt_coefs._add_values(detail, scale)

#     # "effective" j2, used in linear regression
#     j2_eff = int(min(max_level, j2) if j2 is not None else max_level)

#     return WaveletTransform(wt_leaders=wt_leaders,
#                             wt_coefs=wt_coefs,
#                             j2_eff=j2_eff)


def integrate_wavelet(wt_coefs, gamint):
    """
    Fractionally integrates the wavelet coef decomposition of a signal
    """

    if (isinstance(wt_coefs, multiresquantity.WaveletWSE)
            or isinstance(wt_coefs, multiresquantity.WaveletLeader)):
        raise ValueError(
            'Input multi-resolution quantity should be wavelet coef')

    wt_int = multiresquantity.WaveletDec(
        gamint=wt_coefs.gamint + gamint, wt_name=wt_coefs.wt_name,
        n_channel=wt_coefs.n_channel, origin_mrq=wt_coefs)

    for scale in wt_coefs.values:

        wt_int._add_values(
            wt_coefs.values[scale] * 2 ** (gamint * scale), scale)\

    return wt_int


def wavelet_analysis(signal, wt_name='db3', j2=None, normalization=1):
    """
    Compute wavelet coefficient and wavelet leaders.

    Parameters
    ----------
    signal : ndarray of float, shape (n_samples,) | (n_samples, n_realisations)
        Time series to analyze.

    wt_name : str
        Name of the wavelet function to use, as defined in the pywavelet
        package [1]_. The default value of ``'db3'`` means Daubechies with 3
        vanishing moments.

    j2 : int | None
        Upper bound of the scale range for which wavelet coefficients
        will be computed. If None, it will automatically be set to the
        highest value possible.

    normalization : int
        Norm to use on the wavelet coefficients, see notes for more details.

    Returns
    -------
    :class:`.WaveletDec`
        Wavelet coefficient representation of the signal.

    Notes
    -----
    When computing the wavelet coefficients, the values corrupted
    by border effects are set to NaN (np.nan).

    This makes it easier to compute the wavelet leaders, since
    corrupted values will also be nan and can be easily discarded.

    .. note:: Wavelet coefficients are usually L^1 normalized [2]_, which is
              achieved by setting ``normalization=1``.

    References
    ----------

    .. [1] https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html

    .. [2] \
        http://perso-math.univ-mlv.fr/users/jaffard.stephane/pdf/Mandelbrot.pdf
        , page5
    """

    if signal.ndim == 1:
        signal = signal[:, None]

    # if isinstance(gamint, np.ndarray) and gamint.ndim == 1:
    #     gamint = gamint[None, :]

    # Initialize the filter
    wavelet = pywt.Wavelet(wt_name)
    # Investigate why -1
    high_filter = -1*np.array(wavelet.dec_hi)[:, None]
    # low_filter = np.array(wavelet.dec_lo)[:, None]

    max_level = _decomposition_level(signal, len(high_filter), j2)
    approx = signal

    # Initialize structures
    wt_coefs = multiresquantity.WaveletDec(
        gamint=0, wt_name=wt_name, n_channel=signal.shape[1])

    for scale in range(1, max_level + 1):

        # detail shape (N_coef_at_scale, n_realisations)

        detail, approx = _filtering2(approx, wavelet)

        # normalization
        detail = detail*2**(scale*(0.5-1/normalization))

        # fractional integration
        # detail = detail*2.0**(gamint*scale)

        # finite_idx_coef = np.logical_not(np.isinf(np.abs(detail)))
        finite_idx_coef = np.logical_not(np.isnan(np.abs(detail)))

        if np.sum(finite_idx_coef, axis=0).min() < 3:
            max_level = scale-1
            break

        # remove infinite values and store wavelet coefficients
        wt_coefs._add_values(detail, scale)

    if max_level == 0:
        return None

    return wt_coefs


def compute_wse(wt_coefs, theta=0.5, omega=1):
    """
    Compute the (theta,1)-leaders from the wavelet transform.
    """

    wse_coef = multiresquantity.WaveletWSE(
        n_channel=wt_coefs.n_channel, origin_mrq=wt_coefs, wt_name=wt_coefs.wt_name,
        gamint=wt_coefs.gamint, theta=theta)

    for scale, dwt in wt_coefs.values.items():

        # Define a scale slice j/sqrt(j)
        # lower_scale = max(1, int(scale-scale**(theta)))
        lower_scale = max(1, ceil(scale-scale**(theta)))
        # Check the case where j>j_max
        lower_scale = min(lower_scale, scale)
        # J2 = 1  # Dans les cas des leaders J2=1

        wse = np.zeros_like(dwt)  # On initialise le max à 0
        wse.fill(np.nan)

        # On calcule les coefficient l(J1,k)
        for k in range(dwt.shape[0]):

            # On parcourt les tranches d'échelle allant de J2 à J1
            for j in range(scale, lower_scale-1, -1):

                # On prend des paquets de coefficients qui varie
                packet_size = (scale - j) ** omega
                # packet_size = 1  # Pour calculer les leaders

                # On stock tous les coefs en ondelettes
                cwav = wt_coefs.values[j]
                nwav = cwav.shape[0]  # On compte le nombre de coefs

                # On calcule la borne de gauche
                # left_bound = int(
                #     max(1, 2**(scale-j) * (k-packet_size+1))) - 1
                left_bound = max(
                    0, 2**(scale-j) * (k + .5 - packet_size) - .5)

                # On calcule la borne de droite
                right_bound = min(
                    nwav, 2**(scale-j) * (k + .5 + packet_size) - .5 + 1)

                # Calculate indices associated with the bounds
                left_idx = ceil(left_bound)
                right_idx = floor(right_bound)

                # if right_bound <= left_bound:
                #     continue

                # On détermine le WSE

                joint = np.c_[wse[k],
                              np.max(abs(cwav[left_idx:right_idx]), axis=0)]

                # skipping if all nan: wse[k] is already set to nan.
                if np.isnan(joint).all():
                    continue

                wse[k] = np.nanmax(joint, axis=1)

        # print(wse[k].shape)
        # print(np.c_[wse[k],
        #             np.max(abs(cwav[left_bound:right_bound]), axis=0)].shape)

        wse_coef._add_values(wse, scale)

    return wse_coef
