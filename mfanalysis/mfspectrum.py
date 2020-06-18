from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import Utils, smart_power

WaveletParameters = namedtuple('WaveletParameters', 'j1 j2 wtype')


class MultifractalSpectrum:
    """
    This class estimates the multifractal spectrum.
    Based on equations 2.74 - 2.78 of Herwig Wendt's thesis
    (https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf)

    mrq (MultiResolutionQuantity): multiresolution quantity
    q  (numpy.array)             : numpy array containing the exponents q
    wtype (int)                  : 0 for ordinary regression,
                                   1 for weighted regression
    j1 (int)                     : smallest scale analysis
    j2 (int)                     : largest scale analysis

    Dq (np.array)                : y-axis of multifractal spectrum
    hq (np.array)                : x-axis of multifractal spectrum

    """

    def __init__(self, mrq, q, j1, j2, wtype, **kwargs):
        # self.mrq = mrq
        self.nj = mrq.nj
        self.mrq_name = mrq.name
        self.j = np.array(list(mrq.values))
        self.q = q

        self.Dq = None
        self.hq = None

        self.wt_param = WaveletParameters(j1=j1, j2=j2, wtype=wtype)

        # Compute spectrum
        self._compute(mrq)

    def _compute(self, mrq):
        """
        Computes the multifractal spectrum (Dq, hq)
        """

        # Compute U(j,q) and V(j, q)
        U = np.zeros((len(self.j), len(self.q)))
        V = np.zeros((len(self.j), len(self.q)))

        for ind_j, j in enumerate(self.j):
            nj = mrq.nj[j]
            mrq_values_j = np.abs(mrq.values[j])

            # for ind_q, qq in enumerate(self.q):
            #     temp = smart_power(mrq_values_j, qq)  # vector of size nj

            #     R_q_j = temp/temp.sum()

            #     V[ind_j, ind_q] = (R_q_j*np.log2(mrq_values_j)).sum()
            #     U[ind_j, ind_q] = np.log2(nj) + (R_q_j*np.log2(R_q_j)).sum()

            temp = np.stack([smart_power(mrq_values_j, q) for q in self.q])
            R_j = temp / temp.sum(axis=1)[:, None]
            V[ind_j, :] = (R_j * np.log2(mrq_values_j)).sum(axis=1)
            U[ind_j, :] = np.log2(nj) + (R_j * np.log2(R_j)).sum(axis=1)

        self.U = U
        self.V = V

        # Compute D(q) and h(q) via linear regressions
        Dq = np.zeros(len(self.q))
        hq = np.zeros(len(self.q))

        x = np.arange(self.wt_param.j1, self.wt_param.j2+1)

        # weights
        if self.wt_param.wtype == 1:
            wj = self.get_nj_interv(self.wt_param.j1, self.wt_param.j2)
        else:
            wj = np.ones(len(x))

        for ind_q in range(len(self.q)):
            y = U[(self.wt_param.j1-1):self.wt_param.j2, ind_q]
            z = V[(self.wt_param.j1-1):self.wt_param.j2, ind_q]

            slope_1, _ = Utils().linear_regression(x, y, wj)
            slope_2, _ = Utils().linear_regression(x, z, wj)

            Dq[ind_q] = 1 + slope_1
            hq[ind_q] = slope_2

        self.Dq = Dq
        self.hq = hq

    def plot(self, figlabel=None):
        """
        Plot the multifractal spectrum

        figlabel: figure number or name
        """
        if figlabel is None:
            figlabel = 'Multifractal Spectrum'

        plt.figure(figlabel)
        plt.plot(self.hq, self.Dq, 'ko-')
        plt.grid()
        plt.xlabel('h(q)')
        plt.ylabel('D(q)')
        plt.suptitle(self.mrq_name + ' - multifractal spectrum')
        plt.draw()

    def get_nj_interv(self, j1, j2):
        """
        Returns nj as a list, for j in [j1,j2]
        """
        nj = []
        for j in range(j1, j2+1):
            nj.append(self.nj[j])
        return nj
