"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import dataclass, field, InitVar

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as binomial_coefficient
from scipy.stats import trim_mean, median_abs_deviation

from .utils import linear_regression, fast_power
from .multiresquantity import MultiResolutionQuantity, \
    MultiResolutionQuantityBase
# from .viz import plot_multiscale


@dataclass
class Cumulants(MultiResolutionQuantityBase):
    r"""
    Computes and analyzes cumulants

    Parameters
    ----------
    mrq : MultiResolutionQuantity
        Multi resolution quantity to analyze.
    n_cumul : int
        Number of cumulants to compute.
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
    values : ndarray, shape (n_cumulants, n_scales, nrep)
        :math:`C_m(j)`.
    n_cumul : int
        Number of computed cumulants.
    j1 : int
        Lower-bound of the scale support for the linear regressions.
    j2 : int
        Upper-bound of the scale support for the linear regressions.
    weighted : bool
        Whether weighted regression was performed.
    m : ndarray, shape (n_cumulants,)
        List of the m values (cumulants), in order presented in the value
        arrays.
    j : ndarray, shape (n_scales,)
        List of the j values (scales), in order presented in the value arrays.
    log_cumulants : ndarray, shape (n_cumulants, nrep)
        :math:`(c_m)_m`, slopes of the curves :math:`j \times C_m(j)`.
    var_log_cumulants : ndarray, shape (n_cumulants, nrep)
        Estimates of the log-cumulants

        .. warning:: var_log_cumulants
                     was not debugged
    nrep : int
        Number of realisations

    """
    mrq: InitVar[MultiResolutionQuantity]
    n_cumul: int
    j1: int
    j2: int
    weighted: bool
    m: np.ndarray = field(init=False)
    j: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    log_cumulants: np.ndarray = field(init=False)
    var_log_cumulants: np.ndarray = field(init=False)

    def __post_init__(self, mrq):

        self.formalism = mrq.formalism
        self.nj = mrq.nj
        self.nrep = mrq.nrep
        self.j = np.array(list(mrq.values))

        self.m = np.arange(1, self.n_cumul+1)
        self.values = np.zeros((len(self.m), len(self.j), self.nrep))

        self._compute(mrq)
        self._compute_log_cumulants()

    def _compute(self, mrq):

        moments = np.zeros((len(self.m), len(self.j), self.nrep))

        for ind_j, j in enumerate(self.j):

            T_X_j = np.abs(mrq.values[j])
            log_T_X_j = np.log(T_X_j)

            for ind_m, m in enumerate(self.m):

                moments[ind_m, ind_j] = np.nanmean(fast_power(log_T_X_j, m),
                                                   axis=0)
                if m == 1:
                    self.values[ind_m, ind_j] = moments[ind_m, ind_j]
                else:
                    aux = 0

                    for ind_n, n in enumerate(np.arange(1, m)):
                        aux += (binomial_coefficient(m-1, n-1)
                                * self.values[ind_n, ind_j]
                                * moments[ind_m-ind_n-1, ind_j])

                    self.values[ind_m, ind_j] = moments[ind_m, ind_j] - aux

    def _compute_log_cumulants(self):
        """
        Compute the log-cumulants
        (angular coefficients of the curves j->log[C_p(j)])
        """
        self.log_cumulants = np.zeros(((len(self.m), self.nrep)))
        self.var_log_cumulants = np.zeros((len(self.m), self.nrep))
        self.slope = np.zeros((len(self.m), self.nrep))
        self.intercept = np.zeros((len(self.m), self.nrep))

        log2_e = np.log2(np.exp(1))

        x = np.tile(np.arange(self.j1, self.j2+1)[:, None],
                    (1, self.nrep))

        if self.weighted:
            nj = self.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones((len(x), self.nrep))

        # nj = np.tile(nj[:, None], (1, self.nrep))

        ind_j1 = self.j1-1
        ind_j2 = self.j2-1

        for ind_m, _ in enumerate(self.m):
            y = self.values[ind_m, ind_j1:ind_j2+1]
            # pylint: disable=unbalanced-tuple-unpacking
            slope, intercept, var_slope = \
                linear_regression(x, y, nj, return_variance=True)
            self.log_cumulants[ind_m] = slope*log2_e
            self.var_log_cumulants[ind_m] = (log2_e**2)*var_slope
            self.slope[ind_m] = slope
            self.intercept[ind_m] = intercept

    def __getattr__(self, name):

        if name[0] == 'c' and len(name) == 2 and name[1:].isdigit():
            return self.log_cumulants[self.m == int(name[1])]

        if name[0] == 'C' and len(name) == 2 and name[1:].isdigit():
            return self.values[self.m == int(name[1])]

        if name == 'M':
            return -self.c2

        return self.__getattribute__(name)

    def sum(self, cumulants):
        """
        Computes the sum of two cumulants C_m^a(j) and C_m^b(j) weighted by nj:
          C_m^{a+b}(j) = [n_a(j)*C_m^a(j) + n_b(j)*C_m^b(j)]/(n_a(j) + n_b(j))

         Usage:
           cumulants_a.sum(cumulants_b),
            and the result is stored in cumulants_a

        Important:
            * n_cumul, weighted, j1 and j2 must be the same in both objects
            * the attribute cumulants_a.mrq is set to None

        TODO Debug this function
        """

        # Verifications
        assert self.j1 == cumulants.j1, \
            "j1 must be the same for both cumulants"
        assert self.j2 == cumulants.j2, \
            "j2 must be the same for both cumulants"
        assert self.weighted == cumulants.weighted, \
            "weighted must be the same for both cumulants"
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
            for ind_m, _ in enumerate(self.m):
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
        self.j = new_j
        self.nj = new_nj
        self.values = sum_values

        # compute new log_cumulants
        self._compute_log_cumulants()

    def plot(self, fignum=1, nrow=3, filename=None):
        """
        Plots the cumulants.
        Args:
        fignum(int):  figure number
        plt        :  pointer to matplotlib.pyplot
        """

        nrow = min(nrow, len(self.m))

        if len(self.m) > 1:
            plot_dim_1 = nrow
            plot_dim_2 = int(np.ceil(len(self.m) / nrow))

        else:
            plot_dim_1 = 1
            plot_dim_2 = 1

        fig, axes = plt.subplots(plot_dim_1,
                                 plot_dim_2,
                                 num=fignum,
                                 squeeze=False)

        fig.suptitle(self.formalism + r' - cumulants $C_m(j)$')

        x = self.j
        for ind_m, m in enumerate(self.m):
            y = self.values[ind_m, :]

            ax = axes[ind_m % nrow][ind_m // nrow]

            if self.nrep == 1:

                ax.plot(x, y, 'r--.')
                ax.set_xlabel('j')
                ax.set_ylabel('m = ' + str(m))
                # ax.grid()
                # plt.draw()

                if len(self.log_cumulants) > 0:
                    # plot regression line
                    x0 = self.j1
                    x1 = self.j2
                    slope_log2_e = self.log_cumulants[ind_m]
                    slope = self.slope[ind_m]
                    intercept = self.intercept[ind_m]
                    y0 = slope*x0 + intercept
                    y1 = slope*x1 + intercept
                    legend = (r'slope [$\times \log_2(e)]$ = '
                              '%.5f' % (slope_log2_e))

                    ax.plot([x0, x1], [y0, y1], color='k',
                            linestyle='-', linewidth=2, label=legend)
                    ax.legend()
                    plt.draw()

            else:
                pass
            # plot_multiscale({(i, 'cm'): self.values[m, j, nrep] for },
            #                 {'cm': '#00000020', 'cm_avg': '#000000ff'}, ax)

        for j in range(ind_m + 1, len(axes.flat)):
            fig.delaxes(axes[j % nrow][j // nrow])

        if filename is not None:
            plt.savefig(filename)


@dataclass
class MeadCumulants(Cumulants):
    m: np.ndarray = field(init=False, default=np.array([1, 2]))

    def __post_init__(self, mrq):

        self.formalism = mrq.formalism
        self.nj = mrq.nj
        self.j = np.array(list(mrq.values))

        # Force number of cumulants to 2 as it is all we can compute
        self.n_cumul = 2

        self.values = np.zeros((len(self.m), len(self.j)))

        self._compute(mrq)
        self._compute_log_cumulants()

    def _compute(self, mrq):

        for ind_j, j in enumerate(self.j):

            T_X_j = np.abs(mrq.values[j])
            log_T_X_j = np.log(T_X_j)

            self.values[0, ind_j] = \
                np.median(log_T_X_j) * np.log2(np.exp(1))
            self.values[1, ind_j] = \
                (median_abs_deviation(log_T_X_j) ** 2) * np.log2(np.exp(1))

    def _compute_log_cumulants(self):
        """
        Compute the log-cumulants
        (angular coefficients of the curves j->log[C_p(j)])
        """
        self.slope = np.zeros(len(self.m))
        self.intercept = np.zeros(len(self.m))

        self.log_cumulants = np.zeros(len(self.m))
        self.var_log_cumulants = np.zeros(len(self.m))

        log2_e = np.log2(np.exp(1))
        x = np.arange(self.j1, self.j2+1)

        if self.weighted:
            nj = self.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones(len(x))

        ind_j1 = self.j1-1
        ind_j2 = self.j2-1

        for ind_m in range(len(self.m)):

            y = self.values[ind_m, ind_j1:ind_j2+1]
            # pylint: disable=unbalanced-tuple-unpacking
            slope, intercept, var_slope = \
                linear_regression(x, y, nj, return_variance=True)
            # TODO check if slope*log2_e is what is needed here
            self.log_cumulants[ind_m] = slope
            self.var_log_cumulants[ind_m] = (log2_e**2)*var_slope
            self.slope[ind_m] = slope
            self.intercept[ind_m] = intercept


@dataclass
class TrimCumulants(Cumulants):

    def _compute(self, mrq):

        moments = np.zeros((len(self.m), len(self.j)))

        for ind_j, j in enumerate(self.j):

            T_X_j = np.abs(mrq.values[j])
            log_T_X_j = np.log(T_X_j)

            for ind_m, m in enumerate(self.m):

                moments[ind_m, ind_j] = trim_mean(fast_power(log_T_X_j, m),
                                                  0.05)
                if m == 1:
                    self.values[ind_m, ind_j] = moments[ind_m, ind_j]
                else:
                    aux = 0

                    for ind_n, n in enumerate(np.arange(1, m)):
                        aux += (binomial_coefficient(m-1, n-1)
                                * self.values[ind_n, ind_j]
                                * moments[ind_m-ind_n-1, ind_j])

                    self.values[ind_m, ind_j] = \
                        moments[ind_m, ind_j] - aux
