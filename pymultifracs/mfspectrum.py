"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import dataclass, InitVar, field
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


from .ScalingFunction import ScalingFunction
from .regression import linear_regression, prepare_regression, prepare_weights
from .utils import fast_power, fixednansum
from .multiresquantity import MultiResolutionQuantityBase,\
    MultiResolutionQuantity


@dataclass
class MultifractalSpectrum(MultiResolutionQuantityBase, ScalingFunction):
    """
    Estimates the Multifractal Spectrum

    Based on equations 2.74 - 2.78 of Herwig Wendt's thesis [1]_

    Parameters
    ----------
    mrq : MultiResolutionQuantity
        Multi resolution quantity to analyze.
    q : ndarray, shape (n_exponents,)
        Exponents used construct the multifractal spectrum
    j1 : int
        Lower-bound of the scale support for the linear regressions.
    j2 : int
        Upper-bound of the scale support for the linear regressions.
    weighted : str | None
        Whether to used weighted linear regressions.

    Attributes
    ----------
    formalism : str
        Formalism used. Can be any of 'wavelet coefs', 'wavelet leaders',
        or 'wavelet p-leaders'.
    nj : dict(ndarray)
        Number of coefficients at scale j.
        Arrays are of the shape (nrep,)
    j : ndarray, shape (n_scales,)
        List of the j values (scales), in order presented in the value arrays.
    j1 : int
        Lower-bound of the scale support for the linear regressions.
    j2 : int
        Upper-bound of the scale support for the linear regressions.
    weighted : str | None
        Whether weighted regression was performed.
    q : ndarray, shape(n_exponents,)
        Exponents used construct the multifractal spectrum
    Dq : ndarray, shape (n_exponents, nrep)
        Fractal dimensions : :math:`D(q)`, y-axis of the multifractal spectrum
    hq : ndarray, shape (n_exponents, nrep)
        Hölder exponents : :math:`h(q)`, x-axis of the multifractal spectrum
    U : ndarray, shape (n_scales, n_exponents, nrep)
        :math:`U(j, q)`
    V : ndarray, shape (n_scales, n_exponents, nrep)
        :math:`V(j, q)`
    nrep : int
        Number of realisations

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
    bootstrapped_mfs: MultiResolutionQuantityBase = None
    weighted: str = None
    Dq: np.array = field(init=False)
    hq: np.array = field(init=False)
    U: np.array = field(init=False)
    V: np.array = field(init=False)

    def __post_init__(self, mrq):

        self.formalism = mrq.formalism
        self.nj = mrq.nj
        self.nrep = mrq.nrep
        self.j = np.array(list(mrq.values))
        self.bootstrapped_mrq = self.bootstrapped_mfs

        self._compute(mrq)

    def _compute(self, mrq):
        """
        Computes the multifractal spectrum (Dq, hq)
        """

        # 1. Compute U(j,q) and V(j, q)

        # shape (n_q, n_scales, n_rep)
        U = np.zeros((len(self.q), len(self.j), self.nrep))
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

            # if j > 10:
            #     import ipdb; ipdb.set_trace()

        U[np.isinf(U)] = np.nan
        V[np.isinf(V)] = np.nan

        self.U = U
        self.V = V

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j
        )

        # 2. Compute D(q) and h(q) via linear regressions

        # shape (n_q, n_scaling_ranges, n_rep)
        Dq = np.zeros((len(self.q), n_ranges, self.nrep))
        hq = np.zeros_like(Dq)

        # x = np.tile(np.arange(self.j1, self.j2+1)[:, None],
        #             (1, self.nrep))

        # weights
        # if self.weighted:
        #    wj = self.get_nj_interv(self.j1, self.j2)
        # else:
        #    wj = np.ones((len(x), self.nrep))

        # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
        y = U[:, j_min_idx:j_max_idx, None, :]
        z = V[:, j_min_idx:j_max_idx, None, :]

        if self.weighted == 'bootstrap':

            # case where self is the bootstrapped mrq
            if self.bootstrapped_mfs is None:
                std_V = getattr(self, "STD_V")[:, j_min_idx:j_max_idx]
                std_U = getattr(self, "STD_U")[:, j_min_idx:j_max_idx]

            else:
                std_V = getattr(self.bootstrapped_mfs, "STD_V")[:, j_min - self.bootstrapped_mfs.j.min():j_max - self.bootstrapped_mfs.j.min() + 1]
                std_U = getattr(self.bootstrapped_mfs, "STD_U")[:, j_min - self.bootstrapped_mfs.j.min():j_max - self.bootstrapped_mfs.j.min() + 1]

        else:
            std_V = None
            std_U = None

        self.weights = {}

        self.weights['w_V'] = prepare_weights(self, self.weighted, n_ranges, j_min, j_max,
                                              self.scaling_ranges, std_V)
        self.weights['w_U'] = prepare_weights(self, self.weighted, n_ranges, j_min, j_max,
                                              self.scaling_ranges, std_U)

        slope_1, _ = linear_regression(x, y, self.weights['w_U'])
        slope_2, _ = linear_regression(x, z, self.weights['w_V'])

        Dq = 1 + slope_1
        hq = slope_2

        # for ind_q, q in enumerate(self.q):

        #     if self.weighted == 'bootstrap':

        #         # case where self is the bootstrapped mrq
        #         if self.bootstrapped_mfs is None:
        #             std_V = getattr(self, "STD_V_q")(q)[:, j_min-1:j_max]
        #             std_U = getattr(self, "STD_U_q")(q)[:, j_min-1:j_max]

        #         else:
        #             std_V = getattr(self.bootstrapped_mfs, "STD_V_q")(q)[:, j_min-1:j_max]
        #             std_U = getattr(self.bootstrapped_mfs, "STD_U_q")(q)[:, j_min-1:j_max]

        #     else:
        #         std_V = None
        #         std_U = None

        #     self.weights = {}

        #     self.weights['w_V'] = prepare_weights(self, self.weighted, n_ranges, j_min, j_max,
        #                                           self.scaling_ranges, std_V)
        #     self.weights['w_U'] = prepare_weights(self, self.weighted, n_ranges, j_min, j_max,
        #                                           self.scaling_ranges, std_U)

        #     # shape (n_scale, n_scaling_ranges, n_rep)
        #     y = U[j_min-1:j_max, None, ind_q, :]
        #     z = V[j_min-1:j_max, None, ind_q, :]

        #     import ipdb; ipdb.set_trace()
        #     slope_1, _ = linear_regression(x, y, self.weights['w_U'])
        #     slope_2, _ = linear_regression(x, z, self.weights['w_V'])

        #     Dq[ind_q] = 1 + slope_1
        #     hq[ind_q] = slope_2

        self.Dq = Dq
        self.hq = hq

    def V_q(self, q):
        return self.V[self.q == q][0]

    def U_q(self, q):
        return self.U[self.q == q][0]

    def plot(self, figlabel='Multifractal Spectrum', filename=None, ax=None,
             fmt='ko-', scaling_range=0, **plot_kwargs):
        """
        Plot the multifractal spectrum.

        Parameters
        ----------
        figlabel : str
            Figure title
        filename : str | None
            If not None, path used to save the figure
        """

        # plt.figure(figlabel)
        ax = plt.gca() if ax is None else ax

        if self.bootstrapped_mfs is not None:

            CI_Dq = self.CIE_Dq
            CI_hq = self.CIE_hq

            CI_Dq -= self.Dq
            CI_hq -= self.hq

            CI_Dq[:, 1] *= -1
            CI_hq[:, 1] *= -1

            CI_Dq[(CI_Dq < 0) & (CI_Dq > -1e-12)] = 0
            CI_hq[(CI_hq < 0) & (CI_hq > -1e-12)] = 0

            # import ipdb; ipdb.set_trace()

            assert(CI_Dq < 0).sum() == 0
            assert(CI_hq < 0).sum() == 0

            CI_Dq = CI_Dq.transpose(1, 2, 0)
            CI_hq = CI_hq.transpose(1, 2, 0)

        else:
            CI_Dq, CI_hq = None, None

        # print(CI_Dq.shape)
        # print(self.Dq.shape)
        # print(CI_hq.shape)
        # print(self.hq.shape)

        # import ipdb; ipdb.set_trace()

        ax.errorbar(self.hq[:, scaling_range, 0], self.Dq[:, scaling_range, 0], CI_Dq[:, scaling_range], CI_hq[:, scaling_range],
                    fmt, **plot_kwargs)
        ax.set_xlabel('h')
        ax.set_ylabel('D(h)')
        ax.set_ylim((0, 1.05))
        ax.set_xlim((0, 1.5))
        plt.suptitle(self.formalism + ' - multifractal spectrum')
        plt.draw()

        if filename is not None:
            plt.savefig(filename)
