"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import dataclass, InitVar, field
import struct

import numpy as np
import matplotlib.pyplot as plt

from .utils import linear_regression, fast_power
from .multiresquantity import MultiResolutionQuantityBase,\
    MultiResolutionQuantity


@dataclass
class StructureFunction(MultiResolutionQuantityBase):
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
    weighted: bool
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
    weighted : bool
        Whether weighted regression was performed.
    q : ndarray, shape (n_exponents,)
        Exponents for which the structure functions have been computed
    values : ndarray, shape (n_exponents, n_scales, nrep)
        Structure functions : :math:`S(j, q)`
    logvalues : ndarray, shape (n_exponents, n_scales, nrep)
        :math:`\\log_2 S(j, q)`
    zeta : ndarray, shape(n_exponents, nrep)
        Scaling function : :math:`\\zeta(q)`
    H : ndarray, shape (nrep,) | None
        Estimates of H. Set to None if 2 is not in `q`.
    nrep : int
        Number of realisations

    """
    mrq: InitVar[MultiResolutionQuantity]
    q: np.array
    j1: int
    j2: int
    weighted: bool
    j: np.array = field(init=False)
    logvalues: np.array = field(init=False)
    zeta: np.array = field(init=False)
    H: np.array = field(init=False)
    gamint: float = field(init=False)

    def __post_init__(self, mrq):

        self.formalism = mrq.formalism
        self.gamint = mrq.gamint
        self.nrep = mrq.nrep
        self.j = np.array(list(mrq.values))

        self._compute(mrq)
        self._compute_zeta(mrq)
        self.H = self._get_H()

    def _compute(self, mrq):

        values = np.zeros((len(self.q), len(self.j), self.nrep))

        for ind_j, j in enumerate(self.j):

            c_j = mrq.values[j]
            s_j = np.zeros((values.shape[0], self.nrep))

            for ind_q, q in enumerate(self.q):
                s_j[ind_q, :] = np.nanmean(fast_power(np.abs(c_j), q), axis=0)

            values[:, ind_j, :] = s_j

        self.logvalues = np.log2(values)

    def _compute_zeta(self, mrq):
        """
        Compute the value of the scale function zeta(q) for all q
        """
        self.zeta = np.zeros((len(self.q), self.nrep))
        self.intercept = np.zeros((len(self.q), self.nrep))

        x = np.tile(np.arange(self.j1, self.j2+1)[:, None],
                    (1, self.nrep))

        if self.weighted:
            nj = mrq.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones((len(x), self.nrep))

        ind_j1 = self.j1-1
        ind_j2 = self.j2-1
        for ind_q in range(len(self.q)):
            y = self.logvalues[ind_q, ind_j1:ind_j2+1]
            slope, intercept = linear_regression(x, y, nj)
            self.zeta[ind_q] = slope
            self.intercept[ind_q] = intercept

    def _get_H(self):
        return (self.zeta[self.q == 2][0] / 2) - self.gamint

    def S_q(self, q):
        return self.logvalues[self.q == q][0]

    def s_q(self, q):
        return self.zeta[self.q == q][0]

    def get_intercept(self):
        intercept = self.intercept[self.q == 2]

        if len(intercept) > 0:
            return intercept[0]

        return None

    def __getattr__(self, name):

        if name == 'S2':
            return self.logvalues[self.q == 2].squeeze()

        if (super_attr := super().__getattr__(name)) is not None:
            return super_attr

    def plot(self, figlabel='Structure Functions', nrow=4, filename=None,
             ignore_q0=True, figsize=None, struct_boot=None):
        """
        Plots the structure functions.
        """

        if self.nrep > 1:
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

        x = self.j
        counter = 0

        for ind_q, q in enumerate(self.q):

            if q == 0.0 and ignore_q0:
                continue

            y = self.S_q(q)

            if struct_boot is not None:
                CI = struct_boot.CIE_S_q(self)(q)

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
                # plot regression line
                x0 = self.j1
                x1 = self.j2
                slope = self.zeta[ind_q]
                intercept = self.intercept[ind_q]
                y0 = slope*x0 + intercept
                y1 = slope*x1 + intercept

                if struct_boot is not None:
                    CI = struct_boot.CI_s_q(q)
                    CI_legend = f"; [{CI[0]:.3f}, {CI[1]:.3f}]"
                else:
                    CI_legend = ""

                legend = rf'$s_{{{q:.2f}}}$ = {slope[0]:.3f}' + CI_legend

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
        plt.plot(self.q, self.zeta, 'k--.')
        plt.xlabel('q')
        plt.ylabel(r'$\zeta(q)$')
        plt.suptitle(self.formalism + ' - scaling function')

        plt.draw()

        if filename is not None:
            plt.savefig(filename)
