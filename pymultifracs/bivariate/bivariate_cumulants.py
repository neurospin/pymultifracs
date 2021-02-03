# from __future__ import print_function
# from __future__ import unicode_literals

from dataclasses import dataclass, field, InitVar

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as binomial_coefficient

from ..utils import fast_power, linear_regression
from ..multiresquantity import MultiResolutionQuantity, \
    MultiResolutionQuantityBase


@dataclass
class BiCumulants(MultiResolutionQuantityBase):
    mrq1: InitVar[MultiResolutionQuantity]
    mrq2: InitVar[MultiResolutionQuantity]
    n_cumul: int
    j1: int
    j2: int
    wtype: bool
    j: np.array = field(init=False)
    m1: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    log_cumulants: np.ndarray = field(init=False)
    slope: np.ndarray = field(init=False)
    intercept: np.ndarray = field(init=False)
    rho_bMRW: float = field(init=False)
    rho_ss: float = field(init=False)
    rho_mf: float = field(init=False)

    def __post_init__(self, mrq1, mrq2):

        self.nrep = 1

        assert mrq1.formalism == mrq2.formalism
        self.formalism = mrq1.formalism

        assert mrq1.nj == mrq2.nj
        self.nj = mrq1.nj
        self.j = np.array(list(mrq1.values))

        self.m1 = np.arange(0, self.n_cumul+1)
        self.m2 = np.arange(0, self.n_cumul+1)

        self._compute(mrq1, mrq2)
        self._compute_log_cumulants()
        self._compute_rho()

    def _compute(self, mrq1, mrq2):

        moments = np.zeros((self.n_cumul + 1,
                            self.n_cumul + 1,
                            len(self.j)))
        self.values = np.zeros(moments.shape)

        for ind_j, j in enumerate(self.j):

            T_X_j_1 = np.abs(mrq1.values[j])
            T_X_j_2 = np.abs(mrq2.values[j])

            log_T_X_j_1 = np.log(T_X_j_1)
            log_T_X_j_2 = np.log(T_X_j_2)

            for ind_m1, m1 in enumerate(self.m1):
                for ind_m2, m2 in enumerate(self.m2):

                    moments[ind_m1, ind_m2, ind_j] = \
                        np.nanmean((fast_power(log_T_X_j_1, m1))
                                   * fast_power(log_T_X_j_2, m2))

                    if m1 + m2 >= 1:

                        self.values[ind_m1, ind_m2, ind_j] = \
                            (np.nanmean(fast_power(log_T_X_j_1, m1)
                                        * fast_power(log_T_X_j_2, m2))
                             - (self.values[ind_m1, 0, ind_j]
                                * self.values[0, ind_m2, ind_j]))

                    # if m1 == 0:
                    #     if m2 <= 1:
                    #         self.values[ind_m1, ind_m2, ind_j] = \
                    #             moments[ind_m1, ind_m2, ind_j]
                    #     else:

                    #         aux = 0
                    #         for ind_n, n in enumerate(self.m2[:ind_m2-1]):
                    #             aux += (binomial_coefficient(m2-1, n-1)
                    #                     * self.values[ind_m1, ind_n, ind_j]
                    #                     * moments[ind_m1, ind_m2-ind_n-1,
                    #                               ind_j])

                    #         self.values[ind_m1, ind_m2, ind_j] = \
                    #             moments[ind_m1, ind_m2, ind_j] - aux

                    # elif m2 == 0:
                    #     if m1 == 1:
                    #         self.values[ind_m1, ind_m2, ind_j] = \
                    #             moments[ind_m1, ind_m2, ind_j]
                    #     else:

                    #         aux = 0
                    #         for ind_n, n in enumerate(self.m1[:ind_m1-1]):
                    #             aux += (binomial_coefficient(m1-1, n-1)
                    #                     * self.values[ind_n, ind_m2, ind_j]
                    #                     * moments[ind_m1-ind_n-1, ind_m2,
                    #                               ind_j])

                    #         self.values[ind_m1, ind_m2, ind_j] = \
                    #             moments[ind_m1, ind_m2, ind_j] - aux

                    # if m1 == 1 and m2 == 1:

                    #     self.values[ind_m1, ind_m2, ind_j] = \
                    #         (np.nanmean(log_T_X_j_1 * log_T_X_j_2)
                    #          - (self.values[ind_m1, 0, ind_j]
                    #             * self.values[0, ind_m2, ind_j]))

    def _compute_log_cumulants(self):
        """
        Compute the log-cumulants
        (angular coefficients of the curves j->log[C_p(j)])
        """
        self.log_cumulants = np.zeros(self.values.shape[:2])
        self.slope = np.zeros(self.log_cumulants.shape)
        self.intercept = np.zeros(self.log_cumulants.shape)

        log2_e = np.log2(np.exp(1))
        x = np.arange(self.j1, self.j2+1)[:, None]

        if self.wtype:
            nj = self.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones((len(x), 1))

        ind_j1 = self.j1-1
        ind_j2 = self.j2-1

        for ind_m1, _ in enumerate(self.m1):
            for ind_m2, _ in enumerate(self.m2):

                y = self.values[ind_m1, ind_m2, ind_j1:ind_j2+1, None]
                slope, intercept = \
                    linear_regression(x, y, nj, return_variance=False)
                self.log_cumulants[ind_m1, ind_m2] = slope*log2_e
                self.slope[ind_m1, ind_m2] = slope
                self.intercept[ind_m1, ind_m2] = intercept

    def _compute_rho(self):

        self.rho_bMRW =

        self.rho_mf = (- self.log_cumulants[1, 1]
                       / np.abs(np.sqrt(self.log_cumulants[0, 2]
                                        * self.log_cumulants[2, 0])))


class BiCumulants_old:
    """
    This class provides methods for computing and analyzing bivariate
    cumulants, C_{m1, m2}(j)

    For a link between multivariate cumulants and moments, see:
        https://link.springer.com/article/10.1007%2Fs11004-009-9258-9?LI=true
    section 2.1.2

    Args:
        mrq_1 (MultiResolutionQuantity):   multiresolution quantity of signal 1

        mrq_2 (MultiResolutionQuantity):   multiresolution quantity of signal 2

        nj  (dict)  : nj[j] contains the number of coefficients at the scale j

        n_cumul(int): number of cumulants to compute

        m_1 (numpy.array): list of m1 to compute C_{m1, m2}(j) = [|0, n_cumul|]

        m_2 (numpy.array): list of m2 to compute C_{m1, m2}(j)


        j1 (int)                     : smallest scale analysis

        j2 (int)                     : largest scale analysis

        wtype (int)         : 0 for ordinary regression,
                              1 for weighted regression

        values (numpy.array): values[ind_m1, ind_m2, ind_j] =
                                  values of C_{m1, m2}(j),
                              with m_i = self.m_i[ind_mi] and j = self.j[ind_j]

        log_cumulants (numpy.array): slope of the curve  j x C_{m1, m2}(j)
                                         for all (m1, m2)
    """
    def __init__(self, mrq_1, mrq_2, n_cumul, j1, j2, wtype):

        assert n_cumul <= 2, \
            'Bivariate cumulants not implemented for n_cumul > 1'

        self.mrq_1 = mrq_1
        self.mrq_2 = mrq_2

        self.name = mrq_1.name
        self.nj = mrq_1.nj
        self.n_cumul = n_cumul

        self.m_1 = np.arange(0, n_cumul+1)
        self.m_2 = np.arange(0, n_cumul+1)

        self.j1 = j1
        self.j2 = j2
        self.j = np.array(list(mrq_1.values))

        self.wtype = wtype
        self.values = np.zeros((len(self.m_1), len(self.m_2), len(self.j)))
        self.moments = np.zeros((len(self.m_1), len(self.m_2), len(self.j)))
        self.log_cumulants = []
        # self.utils = Utils()  # used for linear regression
        self._compute()
        self._compute_log_cumulants()

    def _compute(self):
        moments = np.zeros((len(self.m_1), len(self.m_2), len(self.j)))

        for ind_j, j in enumerate(self.j):
            T_X_j_1 = np.abs(self.mrq_1.values[j])
            T_X_j_2 = np.abs(self.mrq_2.values[j])

            log_T_X_j_1 = np.log(T_X_j_1)
            log_T_X_j_2 = np.log(T_X_j_2)

            for ind_m1, m1 in enumerate(self.m_1):
                for ind_m2, m2 in enumerate(self.m_2):
                    moments[ind_m1, ind_m2, ind_j] = \
                        np.mean((log_T_X_j_1**m1) * (log_T_X_j_2**m2))

                    # Change here to compute other cumulants
                    # if m1 + m2 >= 1:
                    #     pass

        # Compute C10(j), C01(j) and C11(j)

        self.values = np.zeros((2, 2, len(self.j)))
        # [ [1, C10(j)], [C01(j), C11(j)] ]

        self.values[0, 0, :] = np.ones(len(self.j))
        self.values[1, 0, :] = moments[1, 0, :]
        self.values[0, 1, :] = moments[0, 1, :]

        for ind_j, j in enumerate(self.j):
            T_X_j_1 = np.abs(self.mrq_1.values[j])
            T_X_j_2 = np.abs(self.mrq_2.values[j])
            log_T_X_j_1 = np.log(T_X_j_1)
            log_T_X_j_2 = np.log(T_X_j_2)

            C10j = self.values[1, 0, ind_j]
            C01j = self.values[0, 1, ind_j]
            C11j = np.mean(log_T_X_j_1*log_T_X_j_2) - C01j*C10j
            self.values[1, 1, ind_j] = C11j
        # ----

        self.moments = moments

    def _compute_log_cumulants(self):
        """
        Compute the log-cumulants
        (angular coefficients of the curves j->log[C_p(j)])
        """
        self.log_cumulants = np.zeros((len(self.m_1), len(self.m_2)))
        self.slope = np.zeros((len(self.m_1), len(self.m_2)))
        self.intercept = np.zeros((len(self.m_1), len(self.m_2)))

        log2_e = np.log2(np.exp(1))
        x = np.arange(self.j1, self.j2+1)

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
                    self.utils.linear_regression(x, y, nj,
                                                 return_variance=False)
                self.log_cumulants[ind_m1, ind_m2] = slope*log2_e
                self.slope[ind_m1, ind_m2] = slope
                self.intercept[ind_m1, ind_m2] = intercept

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

    def plot(self, fignum=None):
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
                                 num=fignum,
                                 squeeze=False,
                                 sharex=True)

        fig.suptitle(self.name +
                     r' - bivariate cumulants $\log_2(C_{m_1, m_2}(j))$')

        x = self.j
        plot_index = -1

        for ind_m1, m1 in enumerate(self.m_1):
            for ind_m2, m2 in enumerate(self.m_2):

                if m1+m2 == 0:
                    continue

                plot_index += 1

                y = self.values[ind_m1, ind_m2, :]

                ax = axes[plot_index % 3][plot_index // 3]
                ax.plot(x, y, 'r--.')
                ax.set_xlabel('j')
                ax.set_ylabel(f'$m_1$={m1:1.0f}, $m_2$={m2:1.0f} ')
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

                    legend = r'slope [$\times \log_2(e)]$ = ' + \
                             f'{slope_log2_e:.5f}'

                    ax.plot([x0, x1], [y0, y1], color='k',
                            linestyle='-', linewidth=2, label=legend)
                    ax.legend()
                    plt.draw()
