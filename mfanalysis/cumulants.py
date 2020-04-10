import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as binomial_coefficient

from .utils import Utils


class Cumulants:
    """
    This class provides methods for computing and analyzing cumulants, C_m(j)

    IMPORTANT: var_log_cumulants NOT DEBUGGED

    Args:
        nj  (dict)                   : nj[j] contains the number of
                                       coefficients at the scale j

        n_cumul(int)                 : number of cumulants to compute

        m (numpy.array)              : list of orde m of the cumulant C_m(j)

        j1 (int)                     : smallest scale analysis

        j2 (int)                     : largest scale analysis

        wtype (int)                  : 0 for ordinary regression,
                                       1 for weighted regression

        values (numpy.array)         : values[ind_m, ind_j] = values of C_m(j),
                                       with m = self.m[ind_m]
                                       and j = self.j[ind_j]

        log_cumulants (numpy.array)  : slope of the curve  j x C_m(j) for all m

        var_log_cumulants (numpy.array)  : estimated variance of log-cumulants

    """
    def __init__(self, mrq, n_cumul, j1, j2, wtype, **kwargs):
        self.mrq_name = mrq.name
        self.nj = mrq.nj
        self.n_cumul = n_cumul
        self.m = np.arange(1, n_cumul+1)
        self.j1 = j1
        self.j2 = j2
        self.j = np.array(list(mrq.values))
        self.wtype = wtype
        self.values = np.zeros((len(self.m), len(self.j)))
        self.log_cumulants = []
        self.var_log_cumulants = []
        self.utils = Utils()  # used for linear regression
        self._compute(mrq)
        self._compute_log_cumulants()

    def _compute(self, mrq):

        moments = np.zeros((len(self.m), len(self.j)))

        for ind_j, j in enumerate(self.j):

            T_X_j = np.abs(mrq.values[j])
            log_T_X_j = np.log(T_X_j)

            for ind_m, m in enumerate(self.m):

                moments[ind_m, ind_j] = np.mean(log_T_X_j**m)
                if m == 1:
                    self.values[ind_m, ind_j] = moments[ind_m, ind_j]
                else:
                    aux = 0

                    for ind_n, n in enumerate(np.arange(1, m)):
                        aux += binomial_coefficient(m-1, n-1) * \
                                    self.values[ind_n, ind_j] * \
                                    moments[ind_m-ind_n-1, ind_j]

                    self.values[ind_m, ind_j] = moments[ind_m, ind_j] - aux

    def _compute_log_cumulants(self):
        """
        Compute the log-cumulants
        (angular coefficients of the curves j->log[C_p(j)])
        """
        self.log_cumulants = np.zeros(len(self.m))
        self.var_log_cumulants = np.zeros(len(self.m))
        self.slope = np.zeros(len(self.m))
        self.intercept = np.zeros(len(self.m))

        log2_e = np.log2(np.exp(1))
        x = np.arange(self.j1, self.j2+1)

        if self.wtype == 1:
            nj = self.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones(len(x))

        ind_j1 = self.j1-1
        ind_j2 = self.j2-1
        for ind_m, m in enumerate(self.m):
            y = self.values[ind_m, ind_j1:ind_j2+1]
            slope, intercept, var_slope = \
                self.utils.linear_regression(x, y, nj, return_variance=True)
            self.log_cumulants[ind_m] = slope*log2_e
            self.var_log_cumulants[ind_m] = (log2_e**2)*var_slope
            self.slope[ind_m] = slope
            self.intercept[ind_m] = intercept

    def sum(self, cumulants):
        """
        Computes the sum of two cumulants C_m^a(j) and C_m^b(j) weighted by nj:
          C_m^{a+b}(j) = [n_a(j)*C_m^a(j) + n_b(j)*C_m^b(j)]/(n_a(j) + n_b(j))

         Usage:
           cumulants_a.sum(cumulants_b),
            and the result is stored in cumulants_a

        Important:
            * n_cumul, wtype, j1 and j2 must be the same in both objects
            * the attribute cumulants_a.mrq is set to None
        """

        # Verifications
        assert self.j1 == cumulants.j1, \
            "j1 must be the same for both cumulants"
        assert self.j2 == cumulants.j2, \
            "j2 must be the same for both cumulants"
        assert self.wtype == cumulants.wtype, \
            "wtype must be the same for both cumulants"
        assert self.n_cumul == cumulants.n_cumul, \
            "n_cumul must be the same for both cumulants"

        # max scale
        max_scale_a = self.j.max()
        max_scale_b = cumulants.j.max()
        max_scale = max(max_scale_a, max_scale_b)

        # scales of the result
        new_j = np.arange(1, max_scale+1)

        # compute sum and new nj
        new_nj = {}
        sum_values = np.zeros((len(self.m), len(new_j)))
        for ind_j, j in enumerate(new_j):
            # new nj
            if j <= min(max_scale_a, max_scale_b):
                new_nj[j] = self.nj[j] + cumulants.nj[j]
            elif j > max_scale_a:
                new_nj[j] = cumulants.nj[j]
            elif j > max_scale_b:
                new_nj[j] = self.nj[j]
            # sum
            for ind_m, m in enumerate(self.m):
                if j <= min(max_scale_a, max_scale_b):
                    nj_a = self.nj[j]
                    nj_b = cumulants.nj[j]
                    sum_values[ind_m, ind_j] = \
                        (nj_a*self.values[ind_m, ind_j] +
                         nj_b*cumulants.values[ind_m, ind_j]) / (nj_a+nj_b)

                elif j > max_scale_a:
                    sum_values[ind_m, ind_j] = cumulants.values[ind_m, ind_j]

                elif j > max_scale_b:
                    sum_values[ind_m, ind_j] = self.values[ind_m, ind_j]

        # update self
        self.mrq = None
        self.j = new_j
        self.nj = new_nj
        self.values = sum_values

        # compute new log_cumulants
        self._compute_log_cumulants()

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

        if len(self.m) > 1:
            plot_dim_1 = 3
            plot_dim_2 = int(np.ceil(len(self.m) / 3.0))

        else:
            plot_dim_1 = 1
            plot_dim_2 = 1

        fig, axes = plt.subplots(plot_dim_1,
                                 plot_dim_2,
                                 num=fignum,
                                 squeeze=False)

        fig.suptitle(self.mrq_name + r' - cumulants $\log_2(C_m(j))$')

        x = self.j
        for ind_m, m in enumerate(self.m):
            y = self.values[ind_m, :]

            ax = axes[ind_m % 3][ind_m // 3]
            ax.plot(x, y, 'r--.')
            ax.set_xlabel('j')
            ax.set_ylabel('m = ' + str(m))
            ax.grid()
            plt.draw()

            if len(self.log_cumulants) > 0:
                # plot regression line
                x0 = self.j1
                x1 = self.j2
                slope_log2_e = self.log_cumulants[ind_m]
                slope = self.slope[ind_m]
                intercept = self.intercept[ind_m]
                y0 = slope*x0 + intercept
                y1 = slope*x1 + intercept
                legend = r'slope [$\times \log_2(e)]$ = ' + \
                         f'{slope_log2_e:.5f}'

                ax.plot([x0, x1], [y0, y1], color='k',
                        linestyle='-', linewidth=2, label=legend)
                ax.legend()
                plt.draw()
