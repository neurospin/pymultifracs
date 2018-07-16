from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from .utils import Utils


class MultifractalSpectrum:
    """
    This class estimates the multifractal spectrum.
    Based on equations 2.74 - 2.78 of Herwig Wendt's thesis
    (https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf)

    mrq (MultiResolutionQuantity):   multiresolution quantity 
    q  (numpy.array)             :   numpy array containing the exponents q
    wtype (int)                  :   0 for ordinary regression, 1 for weighted regression
    j1 (int)                     : smallest scale analysis
    j2 (int)                     : largest scale analysis

    Dq (np.array)                : y-axis of multifractal spectrum  
    hq (np.array)                : x-axis of multifractal spectrum

    """

    def __init__(self, mrq, q, j1, j2, wtype):
        self.mrq   = mrq
        self.name  = mrq.name
        self.nj        = mrq.nj
        self.j         = np.array(list(mrq.values))
        self.q     = q
        self.wtype = wtype
        self.j1    = j1
        self.j2    = j2
        self.utils = Utils() # used for linear regression

        self.Dq = None 
        self.hq = None

        # Compute spectrum
        self._compute()

    def _compute(self):
        """
        Computes the multifractal spectrum (Dq, hq)
        """

        # Compute U(j,q) and V(j, q)
        U = np.zeros(( len(self.j), len(self.q)  ))
        V = np.zeros(( len(self.j), len(self.q)  ))
        
        for ind_j, j in enumerate(self.j):
            nj = self.nj[j]
            mrq_values_j = np.abs(self.mrq.values[j])

            for ind_q, qq in enumerate(self.q):
                temp = np.power(mrq_values_j, qq)  # vector of size nj

                R_q_j = temp/temp.sum()

                V[ind_j, ind_q] = (R_q_j*np.log2(mrq_values_j)).sum()
                U[ind_j, ind_q] = np.log2(nj) + (R_q_j*np.log2(R_q_j)).sum()

        self.U = U 
        self.V = V

        # Compute D(q) and h(q) via linear regressions
        Dq = np.zeros(len(self.q))
        hq = np.zeros(len(self.q))

        x  = np.arange(self.j1, self.j2+1)

        # weights
        if self.wtype == 1:
            wj = self.get_nj_interv(self.j1, self.j2)
        else:
            wj = np.ones(len(x))


        for ind_q, q in enumerate(self.q):
            y = U[(self.j1-1):self.j2, ind_q]
            z = V[(self.j1-1):self.j2, ind_q]

            slope_1, intercept_1  = self.utils.linear_regression(x, y, wj) 
            slope_2, intercept_2  = self.utils.linear_regression(x, z, wj) 

            Dq[ind_q] = 1 + slope_1
            hq[ind_q] = slope_2

        self.Dq = Dq
        self.hq = hq

    def plot(self, figlabel = None):
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
        plt.suptitle(self.name + ' - multifractal spectrum')
        plt.draw()


    def get_nj_interv(self, j1, j2):
        """
        Returns nj as a list, for j in [j1,j2]
        """
        nj = []
        for j in range(j1, j2+1):
            nj.append(self.nj[j])
        return nj
