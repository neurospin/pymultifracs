"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import InitVar, dataclass, field
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .multiresquantity import (MultiResolutionQuantity,
                               MultiResolutionQuantityBase)
from .regression import linear_regression, prepare_regression, prepare_weights
from .ScalingFunction import ScalingFunction
from .utils import MFractalVar, fast_power, fixednansum, isclose


@dataclass
class MultifractalSpectrum(MultiResolutionQuantityBase, ScalingFunction):
    """
    Estimates the Multifractal Spectrum

    Based on equations 2.74 - 2.78 of Herwig Wendt's thesis [1]_

    Parameters
    ----------
    mrq : MultiResolutionQuantity
        Multi resolution quantity to analyze.
    scaling_ranges: List[Tuple[int]]
        List of pairs of (j1, j2) ranges of scales for the analysis
    q : ndarray, shape (n_exponents,)
        Exponents used construct the multifractal spectrum
    boostrapped_mfa: MFractalVar | None
        Output of the MFA of bootstrapped MRQs.
    weighted : str | None
        Whether to used weighted linear regressions.

    Attributes
    ----------
    formalism : str
        Formalism used. Can be any of 'wavelet coefs', 'wavelet leaders',
        or 'wavelet p-leaders'.
    j : ndarray, shape (n_scales,)
        List of the j values (scales), in order presented in the value arrays.
    scaling_ranges: List[Tuple[int]]
        List of pairs of (j1, j2) ranges of scales for the analysis
    weighted : str | None
        If not None, indicates the weighting approach used for regression
    q : ndarray, shape(n_exponents,)
        Exponents used construct the multifractal spectrum
    Dq : ndarray, shape (n_exponents, n_rep)
        Fractal dimensions : :math:`D(q)`, y-axis of the multifractal spectrum
    hq : ndarray, shape (n_exponents, n_rep)
        Hölder exponents : :math:`h(q)`, x-axis of the multifractal spectrum
    U : ndarray, shape (n_scales, n_exponents, n_rep)
        :math:`U(j, q)`
    V : ndarray, shape (n_scales, n_exponents, n_rep)
        :math:`V(j, q)`

    References
    ----------
    .. [1]  H. Wendt (2008). Contributions of Wavelet Leaders and Bootstrap to
        Multifractal Analysis: Images, Estimation Performance, Dependence
        Structure and Vanishing Moments. Confidence Intervals and Hypothesis
        Tests. Ph.D thesis, Laboratoire de Physique, Ecole Normale Superieure
        de Lyon.
        https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf
    """
    mrq: InitVar[MultiResolutionQuantity]
    j: np.array = field(init=False)
    scaling_ranges: List[Tuple[int]]
    q: np.array
    bootstrapped_mfa: InitVar[MFractalVar] = None
    weighted: str = None
    Dq: np.array = field(init=False)
    hq: np.array = field(init=False)
    U: np.array = field(init=False)
    V: np.array = field(init=False)

    def __post_init__(self, mrq, bootstrapped_mfa):

        self.formalism = mrq.formalism
        self.nj = mrq.nj
        self.n_sig = mrq.n_sig
        self.j = np.array(list(mrq.values))

        if bootstrapped_mfa is not None:
            self.bootstrapped_mrq = bootstrapped_mfa.spectrum

        self._compute(mrq)

    def _compute(self, mrq):
        """
        Computes the multifractal spectrum (Dq, hq)
        """

        # 1. Compute U(j,q) and V(j, q)

        # shape (n_q, n_scales, n_rep)
        U = np.zeros((len(self.q), len(self.j), mrq.n_rep))
        V = np.zeros_like(U)

        for ind_j, j in enumerate(self.j):

            nj = mrq.nj[j]
            mrq_values_j = np.abs(mrq.values[j])

            idx_nan = np.isnan(mrq_values_j)
            temp = np.stack([fast_power(mrq_values_j, q) for q in self.q],
                            axis=0)

            # np.nan ** 0 = 1.0, adressed here
            temp[:, idx_nan] = np.nan
            Z = np.nansum(temp, axis=1)[:, None, :]
            Z[Z == 0] = np.nan
            R_j = temp / Z
            V[:, ind_j, :] = fixednansum(R_j * np.log2(mrq_values_j), axis=1)
            U[:, ind_j, :] = np.log2(nj) + fixednansum((R_j * np.log2(R_j)),
                                                       axis=1)

        U[np.isinf(U)] = np.nan
        V[np.isinf(V)] = np.nan

        self.U = U
        self.V = V

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j
        )

        # 2. Compute D(q) and h(q) via linear regressions

        # shape (n_q, n_scaling_ranges, n_rep)
        Dq = np.zeros((len(self.q), n_ranges, mrq.n_rep))
        hq = np.zeros_like(Dq)

        # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
        y = U[:, j_min_idx:j_max_idx, None, :]
        z = V[:, j_min_idx:j_max_idx, None, :]

        if self.weighted == 'bootstrap':

            # case where self is the bootstrapped mrq
            if self.bootstrapped_mrq is None:
                std_V = getattr(self, "STD_V")[:, j_min_idx:j_max_idx]
                std_U = getattr(self, "STD_U")[:, j_min_idx:j_max_idx]

            else:
                std_V = getattr(self.bootstrapped_mrq, "STD_V")[
                    :,
                    j_min - self.bootstrapped_mrq.j.min():
                    j_max - self.bootstrapped_mrq.j.min() + 1]
                std_U = getattr(self.bootstrapped_mrq, "STD_U")[
                    :,
                    j_min - self.bootstrapped_mrq.j.min():
                    j_max - self.bootstrapped_mrq.j.min() + 1]

        else:
            std_V = None
            std_U = None

        self.weights = {}

        self.weights['w_V'] = prepare_weights(
            self, self.weighted, n_ranges, j_min, j_max, self.scaling_ranges,
            std_V)
        self.weights['w_U'] = prepare_weights(
            self, self.weighted, n_ranges, j_min, j_max, self.scaling_ranges,
            std_U)

        slope_1, _ = linear_regression(x, y, self.weights['w_U'])
        slope_2, _ = linear_regression(x, z, self.weights['w_V'])

        Dq = 1 + slope_1
        hq = slope_2

        self.Dq = Dq
        self.hq = hq

    def V_q(self, q):
        out = self.V[isclose(q, self.q)][0]
        return out.reshape(out.shape[0], self.n_sig, -1)

    def U_q(self, q):
        out = self.U[np.isclose(q, self.q)][0]
        return out.reshape(out.shape[0], self.n_sig, -1)

    def __getattr__(self, name):

        if name == 'n_rep':
            return self.Dq.shape[-1]

        return super().__getattr__(name)

    def plot(self, figlabel='Multifractal Spectrum', filename=None, ax=None,
             fmt='ko-', scaling_range=0, signal_idx=0, **plot_kwargs):
        """
        Plot the multifractal spectrum.

        Parameters
        ----------
        figlabel : str
            Figure title
        filename : str | None
            If not None, path used to save the figure
        """

        ax = plt.gca() if ax is None else ax

        if self.bootstrapped_mrq is not None:

            CI_Dq = self.CIE_Dq
            CI_hq = self.CIE_hq

            CI_Dq -= self.Dq
            CI_hq -= self.hq

            CI_Dq[:, 1] *= -1
            CI_hq[:, 1] *= -1

            CI_Dq[(CI_Dq < 0) & (CI_Dq > -1e-12)] = 0
            CI_hq[(CI_hq < 0) & (CI_hq > -1e-12)] = 0

            assert(CI_Dq < 0).sum() == 0
            assert(CI_hq < 0).sum() == 0

            CI_Dq = CI_Dq.transpose(1, 2, 0)
            CI_hq = CI_hq.transpose(1, 2, 0)

            CI_Dq = CI_Dq[:, scaling_range]
            CI_hq = CI_hq[:, scaling_range]

        else:
            CI_Dq, CI_hq = None, None

        ax.errorbar(self.hq[:, scaling_range, 0], self.Dq[:, scaling_range, 0],
                    CI_Dq, CI_hq, fmt,
                    **plot_kwargs)

        ax.set(xlabel='h', ylabel='D(h)', ylim=(0, 1.05), xlim=(0, 1.5))

        plt.suptitle(self.formalism + ' - multifractal spectrum')
        plt.draw()

        if filename is not None:
            plt.savefig(filename)
