from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as binomial_coefficient
from .utils import Utils

class BiCumulants:
    """
    This class provides methods for computing and analyzing bivariate cumulants, C_{m1, m2}(j)

    For a link between multivariate cumulants and moments, see:
        https://link.springer.com/article/10.1007%2Fs11004-009-9258-9?LI=true  section 2.1.2

    Args:
        mrq_1 (MultiResolutionQuantity):   multiresolution quantity of signal 1

        mrq_2 (MultiResolutionQuantity):   multiresolution quantity of signal 2

        nj  (dict)                   : nj[j] contains the number of coefficients at the scale j

        n_cumul(int)                 : number of cumulants to compute

        m_1 (numpy.array)             : list of m1 to compute C_{m1, m2}(j) = [0, ..., n_cumul]

        m_2 (numpy.array)             : list of m2 to compute C_{m1, m2}(j)


        j1 (int)                     : smallest scale analysis

        j2 (int)                     : largest scale analysis

        wtype (int)                  : 0 for ordinary regression, 1 for weighted regression

        values (numpy.array)         : values[ind_m1, ind_m2, ind_j] = values of C_{m1, m2}(j), with m_i = self.m_i[ind_mi]
                                         and j = self.j[ind_j]

        log_cumulants (numpy.array)  : slope of the curve  j x C_{m1, m2}(j) for all (m1, m2)
    """
    def __init__(self, mrq_1, mrq_2 , n_cumul, j1, j2, wtype):

        assert n_cumul <= 2, 'Bivariate cumulants not implemented for n_cumul > 1'

        self.mrq_1     = mrq_1
        self.mrq_2     = mrq_2

        self.name      = mrq_1.name
        self.nj        = mrq_1.nj
        self.n_cumul   = n_cumul
        self.m_1         = np.arange(0, n_cumul+1)
        self.m_2         = np.arange(0, n_cumul+1)

        self.j1        = j1
        self.j2        = j2
        self.j         = np.array(list(mrq_1.values))
        self.wtype     = wtype
        self.values    = np.zeros( (len(self.m_1), len(self.m_2) , len(self.j))  )
        self.moments   = np.zeros( (len(self.m_1), len(self.m_2) , len(self.j))  )
        self.log_cumulants  = []
        self.utils = Utils() # used for linear regression
        self._compute()
        self._compute_log_cumulants()

    def _compute(self):
        moments = np.zeros( (len(self.m_1), len(self.m_2) ,len(self.j))  )

        for ind_j, j in enumerate(self.j):
            T_X_j_1 = np.abs(self.mrq_1.values[j])
            T_X_j_2 = np.abs(self.mrq_2.values[j])

            log_T_X_j_1 = np.log(T_X_j_1)
            log_T_X_j_2 = np.log(T_X_j_2)


            for ind_m1, m1 in enumerate(self.m_1):
                for ind_m2, m2 in enumerate(self.m_2):
                    moments[ind_m1, ind_m2 ,ind_j] =  np.mean(  (log_T_X_j_1**m1)* (log_T_X_j_2**m2)  )

                    # Change here to compute other cumulants
                    # if m1 + m2 >= 1:
                    #     pass

        # Compute C10(j), C01(j) and C11(j)
        self.values = np.zeros((2, 2, len(self.j))) # [ [1, C10(j)], [C01(j), C11(j)] ]
        self.values[0, 0, :] = np.ones(len(self.j))
        self.values[1, 0, :] = moments[1, 0, :]
        self.values[0, 1, :] = moments[0, 1, :]

        for ind_j, j in enumerate(self.j):
            T_X_j_1 = np.abs(self.mrq_1.values[j])
            T_X_j_2 = np.abs(self.mrq_2.values[j])

            C10j = self.values[1, 0, :]
            C01j = self.values[0, 1, :]
            C11j = np.mean( log_T_X_j_1*log_T_X_j_2 ) - C01j*C10j
            self.values[1, 1, :] = C11j
        # ----

        self.moments = moments

    def _compute_log_cumulants(self):
        """
        Compute the log-cumulants (angular coefficients of the curves j->log[C_p(j)])
        """
        self.log_cumulants = np.zeros((len(self.m_1), len(self.m_2)))
        self.slope         = np.zeros((len(self.m_1), len(self.m_2)))
        self.intercept     = np.zeros((len(self.m_1), len(self.m_2)))

        log2_e = np.log2(np.exp(1))
        x  = np.arange(self.j1, self.j2+1)

        if self.wtype == 1:
            nj = self.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones(len(x))


        ind_j1 = self.j1-1
        ind_j2 = self.j2-1
        for ind_m1, m1 in enumerate(self.m_1):
            for ind_m2, m2 in enumerate(self.m_2): 
                y = self.values[ind_m1, ind_m2, ind_j1:ind_j2+1]
                slope, intercept = \
                    self.utils.linear_regression(x, y, nj, return_variance = False)
                self.log_cumulants[ind_m1, ind_m2] = slope*log2_e
                self.slope[ind_m1, ind_m2]         = slope
                self.intercept[ind_m1, ind_m2]     = intercept


    def get_nj(self):
        """
        Returns nj as a list
        """
        return list(self.nj.values())

    def get_nj_interv(self, j1, j2):
        """
        Returns nj as a list, for j in [j1,j2]
        """
        nj = []
        for j in range(j1, j2+1):
            nj.append(self.nj[j])
        return nj


    def plot(self, fignum = None):
        """
        Plots the cumulants.
        Args:
            fignum(int):  figure number
            plt        :  pointer to matplotlib.pyplot
        """
        if fignum is None:
            fignum = 1

        nm = len(self.m_1)*len(self.m_2) - 1

        if nm > 1:
            plot_dim_1 = 3
            plot_dim_2 = int(np.ceil(nm / 3.0))

        else:
            plot_dim_1 = 1
            plot_dim_2 = 1

        fig, axes = plt.subplots(plot_dim_1,
            plot_dim_2,
            num = fignum,
            squeeze = False,
            sharex = True)

        fig.suptitle(self.name + ' - bivariate cumulants $\log_2(C_{m_1, m_2}(j))$')

        x = self.j
        plot_index = -1
        for ind_m1, m1 in enumerate(self.m_1):
            for ind_m2, m2 in enumerate(self.m_2):
                if m1+m2 == 0:
                    continue

                plot_index += 1

                y = self.values[ind_m1, ind_m2, :]

                ax  = axes[plot_index % 3][plot_index // 3]
                ax.plot(x, y, 'r--.')
                ax.set_xlabel('j')
                ax.set_ylabel('$m_1$=%d, $m_2$=%d '%(m1,m2))
                ax.grid()
                plt.draw()

                if len(self.log_cumulants) > 0:
                    # plot regression line
                    x0 = self.j1
                    x1 = self.j2 
                    slope_log2_e = self.log_cumulants[ind_m1, ind_m2]
                    slope = self.slope[ind_m1, ind_m2]
                    intercept = self.intercept[ind_m1, ind_m2]
                    y0 = slope*x0 + intercept
                    y1 = slope*x1 + intercept
                    legend = 'slope [$\\times \log_2(e)]$ = '+'%.5f' % (slope_log2_e)

                    ax.plot([x0, x1], [y0, y1], color='k',
                        linestyle='-', linewidth=2, label = legend)
                    ax.legend()
                    plt.draw()
