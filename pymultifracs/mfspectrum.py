"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import dataclass, InitVar, field

import numpy as np
import matplotlib.pyplot as plt

from .utils import linear_regression, fast_power, fixednansum
from .multiresquantity import MultiResolutionQuantityBase,\
    MultiResolutionQuantity


@dataclass
class MultifractalSpectrum(MultiResolutionQuantityBase):
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
    wtype : bool
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
    wtype : bool
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
    j1: int
    j2: int
    wtype: bool
    q: np.array
    Dq: np.array = field(init=False)
    hq: np.array = field(init=False)
    U: np.array = field(init=False)
    V: np.array = field(init=False)

    def __post_init__(self, mrq):

        self.formalism = mrq.formalism
        self.nj = mrq.nj
        self.nrep = mrq.nrep
        self.j = np.array(list(mrq.values))

        self._compute(mrq)

    def _compute(self, mrq):
        """
        Computes the multifractal spectrum (Dq, hq)
        """

        # Compute U(j,q) and V(j, q)
        U = np.zeros((len(self.j), len(self.q), self.nrep))
        V = np.zeros((len(self.j), len(self.q), self.nrep))

        for ind_j, j in enumerate(self.j):
            nj = mrq.nj[j]
            mrq_values_j = np.abs(mrq.values[j])

            ixd_nan = np.isnan(mrq_values_j)
            temp = np.stack([fast_power(mrq_values_j, q) for q in self.q],
                            axis=0)
            # np.nan ** 0 = 1.0, adressed here
            temp[:, ixd_nan] = np.nan
            R_j = temp / np.nansum(temp, axis=1)[:, None, :]
            V[ind_j, :] = fixednansum(R_j * np.log2(mrq_values_j), axis=1)
            U[ind_j, :] = np.log2(nj) + fixednansum((R_j * np.log2(R_j)),
                                                    axis=1)

        self.U = U
        self.V = V

        # Compute D(q) and h(q) via linear regressions
        Dq = np.zeros((len(self.q), self.nrep))
        hq = np.zeros((len(self.q), self.nrep))

        x = np.tile(np.arange(self.j1, self.j2+1)[:, None],
                    (1, self.nrep))

        # weights
        if self.wtype:
            wj = self.get_nj_interv(self.j1, self.j2)
        else:
            wj = np.ones((len(x), self.nrep))

        for ind_q in range(len(self.q)):
            y = U[(self.j1-1):self.j2, ind_q]
            z = V[(self.j1-1):self.j2, ind_q]

            slope_1, _ = linear_regression(x, y, wj)
            slope_2, _ = linear_regression(x, z, wj)

            Dq[ind_q] = 1 + slope_1
            hq[ind_q] = slope_2

        self.Dq = Dq
        self.hq = hq

    def plot(self, figlabel='Multifractal Spectrum', filename=None):
        """
        Plot the multifractal spectrum.

        Parameters
        ----------
        figlabel : str
            Figure title
        filename : str | None
            If not None, path used to save the figure
        """

        plt.figure(figlabel)
        plt.plot(self.hq, self.Dq, 'ko-')
        plt.xlabel('h')
        plt.ylabel('D(h)')
        plt.suptitle(self.formalism + ' - multifractal spectrum')
        plt.draw()

        if filename is not None:
            plt.savefig(filename)
