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
from .utils import fast_power, MFractalVar, isclose
from .multiresquantity import MultiResolutionQuantityBase,\
    MultiResolutionQuantity


@dataclass
class StructureFunction(MultiResolutionQuantityBase, ScalingFunction):
    """
    Computes and analyzes structure functions

    Parameters
    ----------
    mrq : MultiResolutionQuantity
        Multi resolution quantity to analyze.
    q : ndarray, shape (n_exponents,)
        Exponent for which to compute the structure function
    j1 : int
        Lower-bound of the scale support for the linear regressions.
    j2 : int
        Upper-bound of the scale support for the linear regressions.
    weighted: str | None
        Whether to used weighted linear regressions.

    Attributes
    ----------
    formalism : str
        Formalism used. Can be any of 'wavelet coefs', 'wavelet leaders',
        or 'wavelet p-leaders'.
    nj : dict(ndarray)
        Number of coefficients at scale j.
        Arrays are of the shape (n_rep,)
    j : ndarray, shape (n_scales,)
        List of the j values (scales), in order presented in the value arrays.
    j1 : int
        Lower-bound of the scale support for the linear regressions.
    j2 : int
        Upper-bound of the scale support for the linear regressions.
    weighted : str | None
        Whether weighted regression was performed.
    q : ndarray, shape (n_exponents,)
        Exponents for which the structure functions have been computed
    values : ndarray, shape (n_exponents, n_scales, n_rep)
        Structure functions : :math:`S(j, q)`
    logvalues : ndarray, shape (n_exponents, n_scales, n_rep)
        :math:`\\log_2 S(j, q)`
    zeta : ndarray, shape(n_exponents, n_rep)
        Scaling function : :math:`\\zeta(q)`
    H : ndarray, shape (n_rep,) | None
        Estimates of H. Set to None if 2 is not in `q`.
    n_rep : int
        Number of realisations

    """
    mrq: InitVar[MultiResolutionQuantity]
    q: np.array
    scaling_ranges: List[Tuple[int]]
    weighted: str = None
    bootstrapped_mfa: InitVar[MFractalVar] = None
    j: np.array = field(init=False)
    logvalues: np.array = field(init=False)
    zeta: np.array = field(init=False)
    H: np.array = field(init=False)
    gamint: float = field(init=False)

    def __post_init__(self, mrq, bootstrapped_mfa):

        self.formalism = mrq.formalism
        self.gamint = mrq.gamint
        self.n_sig = mrq.n_sig
        self.nj = mrq.nj
        self.j = np.array(list(mrq.values))

        if bootstrapped_mfa is not None:
            self.bootstrapped_mrq = bootstrapped_mfa.structure

        self._compute(mrq)
        self._compute_zeta(mrq.n_rep)
        self.H = self._get_H()

    def _compute(self, mrq):

        values = np.zeros((len(self.q), len(self.j), mrq.n_rep))

        for ind_j, j in enumerate(self.j):

            c_j = mrq.values[j]
            s_j = np.zeros((values.shape[0], mrq.n_rep))

            for ind_q, q in enumerate(self.q):
                s_j[ind_q, :] = np.nanmean(fast_power(np.abs(c_j), q), axis=0)

            values[:, ind_j, :] = s_j

        self.logvalues = np.log2(values)

        self.logvalues[np.isinf(self.logvalues)] = np.nan

    def _compute_zeta(self, n_rep):
        """
        Compute the value of the scale function zeta(q) for all q
        """

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j)

        # shape (n_moment, n_scaling_ranges, n_rep)
        self.zeta = np.zeros((len(self.q), n_ranges, n_rep))
        self.intercept = np.zeros_like(self.zeta)

        # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
        y = self.logvalues[:, j_min_idx:j_max_idx, None, :]

        if self.weighted == 'bootstrap':

            # case where self is the bootstrapped mrq
            if self.bootstrapped_mrq is None:
                std = self.STD_logvalues[:, j_min_idx:j_max_idx]

            else:
                std = self.bootstrapped_mrq.STD_logvalues[
                    :,
                    j_min - self.bootstrapped_mrq.j.min():
                    j_max - self.bootstrapped_mrq.j.min() + 1]

        else:
            std = None

        self.weights = prepare_weights(self, self.weighted, n_ranges, j_min,
                                       j_max, self.scaling_ranges, std)

        self.zeta, self.intercept = linear_regression(x, y, self.weights)

    def compute_R(self):

        values = self.logvalues.reshape(
            *self.logvalues.shape[:2], self.n_sig, -1)
        slope = self.zeta.reshape(*self.zeta.shape[:2], self.n_sig, -1)
        intercept = self.intercept.reshape(
            *self.intercept.shape[:2], self.n_sig, -1)

        return super()._compute_R(values, slope, intercept)

    def compute_R2(self):
        return super()._compute_R2(self.logvalues, self.zeta, self.intercept,
                                   self.weights)

    def _get_H(self):
        return (self.zeta[self.q == 2][0] / 2) - self.gamint

    def S_q(self, q):

        out = self.logvalues[isclose(q, self.q)][0]
        out = out.reshape(out.shape[0], self.n_sig, -1)

        return out

    def s_q(self, q):

        out = self.zeta[isclose(q, self.q)][0]
        out = out.reshape(out.shape[0], self.n_sig, -1)

        return out

    def get_intercept(self):
        intercept = self.intercept[self.q == 2]

        if len(intercept) > 0:
            return intercept[0]

        return None

    def __getattr__(self, name):

        if name == 'S2':
            out = self.logvalues[self.q == 2]
            return out.reshape(out.shape[0], self.n_sig, -1)

        if name == 'n_rep':
            return self.zeta.shape[-1]

        if (super_attr := super().__getattr__(name)) is not None:
            return super_attr

        return self.__getattribute__(name)

    def plot(self, figlabel='Structure Functions', nrow=4, filename=None,
             ignore_q0=True, figsize=None, scaling_range=0, plot_scales=None,
             plot_CI=True):
        """
        Plots the structure functions.
        """

        if self.n_rep > 1:
            raise ValueError('Cannot plot structure functions for more than '
                             '1 repetition at a time')

        nrow = min(nrow, len(self.q))
        nq = len(self.q) + (-1 if 0.0 in self.q and ignore_q0 else 0)

        if nq > 1:
            plot_dim_1 = nrow
            plot_dim_2 = int(np.ceil(nq / nrow))

        else:
            plot_dim_1 = 1
            plot_dim_2 = 1

        fig, axes = plt.subplots(plot_dim_1,
                                 plot_dim_2,
                                 num=figlabel,
                                 squeeze=False,
                                 figsize=figsize)

        fig.suptitle(self.formalism +
                     r' - structure functions $\log_2(S(j,q))$')

        if plot_scales is None:
            idx = np.s_[:]
        else:
            j_min = self.j.min()
            idx = np.s_[plot_scales[0] - j_min:plot_scales[1] - j_min + 1]

        x = self.j[idx]

        counter = 0

        for ind_q, q in enumerate(self.q):

            if q == 0.0 and ignore_q0:
                continue

            y = self.S_q(q)[idx]

            if self.bootstrapped_mrq is not None and plot_CI is not None:

                CI = self.CIE_S_q(q)[idx]

                CI -= y
                CI[:, 1] *= -1
                assert (CI < 0).sum() == 0
                CI = CI.transpose()

            else:
                CI = None

            ax = axes[counter % nrow][counter // nrow]
            ax.errorbar(x, y[:, 0], CI, fmt='r--.', zorder=-1)
            ax.set_xlabel('j')
            ax.set_ylabel(f'q = {q:.3f}')

            counter += 1

            if len(self.zeta) > 0:

                x0, x1 = self.scaling_ranges[scaling_range]
                slope = self.zeta[ind_q, scaling_range, 0]
                intercept = self.intercept[ind_q, scaling_range, 0]

                assert x0 in x, "Scaling range not included in plotting range"
                assert x1 in x, "Scaling range not included in plotting range"

                y0 = slope*x0 + intercept
                y1 = slope*x1 + intercept

                if self.bootstrapped_mrq is not None:
                    CI = self.CIE_s_q(q)[scaling_range]
                    CI_legend = f"; [{CI[0]:.1f}, {CI[1]:.1f}]"
                else:
                    CI_legend = ""

                legend = rf'$s_{{{q:.2f}}}$ = {slope:.2f}' + CI_legend

                ax.plot([x0, x1], [y0, y1], color='k',
                        linestyle='-', linewidth=2, label=legend, zorder=0)
                ax.legend()

        for j in range(counter, len(axes.flat)):
            fig.delaxes(axes[j % nrow][j // nrow])

        plt.draw()

        if filename is not None:
            plt.savefig(filename)

    def plot_scaling(self, figlabel='Scaling Function', filename=None):

        assert len(self.q) > 1, ("This plot is only possible if more than 1 q",
                                 " value is used")

        plt.figure(figlabel)
        plt.plot(self.q, self.zeta[:, 0, 0], 'k--.')
        plt.xlabel('q')
        plt.ylabel(r'$\zeta(q)$')
        plt.suptitle(self.formalism + ' - scaling function')

        plt.draw()

        if filename is not None:
            plt.savefig(filename)
