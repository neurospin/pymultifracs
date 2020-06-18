import numpy as np
import matplotlib.pyplot as plt

from .utils import Utils, smart_power


class StructureFunction:
    """
    This class provides methods for computing and analyzing struture functions

    Args:
        mrq (MultiResolutionQuantity):   multiresolution quantity used to
                                         compute the structure function

        q (numpy.array)              :   list of exponents for which to compute
                                         the structure function

        j1 (int): smallest scale analysis

        j2 (int): largest scale analysis

        wtype (int)            : 0 for ordinary regression,
                                 1 for weighted regression

        values (numpy.array)   : values[ind_q, ind_j] = values of S(j, q),
                                 with q = self.q[ind_q]
                                 and j = self.j[ind_j]

        logvalues (numpy.array): logvalues[ind_q, ind_j] = log_2 (S(j, q)),
                                 with q = self.q[ind_q]  and j = self.j[ind_j]

    """
    def __init__(self, mrq, q, j1, j2, wtype, **kwargs):
        self.mrq_name = mrq.name
        self.q = q
        self.j1 = j1
        self.j2 = j2
        self.j = np.array(list(mrq.values))
        self.wtype = wtype
        self.logvalues = np.zeros((len(self.q), len(self.j)))
        self.zeta = []
        self.utils = Utils()  # used for linear regression
        self._compute(mrq)
        self._compute_zeta(mrq)

    def _compute(self, mrq):

        values = np.zeros((len(self.q), len(self.j)))

        for ind_j, j in enumerate(self.j):
            c_j = mrq.values[j]
            s_j = np.zeros(values.shape[0])
            for ind_q, q in enumerate(self.q):
                s_j[ind_q] = np.mean(smart_power(np.abs(c_j), q))
            values[:, ind_j] = s_j

        self.logvalues = np.log2(values)

    def _compute_zeta(self, mrq):
        """
        Compute the value of the scale function zeta(q) for all q
        """
        self.zeta = np.zeros(len(self.q))
        self.intercept = np.zeros(len(self.q))

        x = np.arange(self.j1, self.j2+1)

        if self.wtype == 1:
            nj = mrq.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones(len(x))

        ind_j1 = self.j1-1
        ind_j2 = self.j2-1
        for ind_q in range(len(self.q)):
            y = self.logvalues[ind_q, ind_j1:ind_j2+1]
            slope, intercept = self.utils.linear_regression(x, y, nj)
            self.zeta[ind_q] = slope
            self.intercept[ind_q] = intercept

    def get_H(self):
        H = self.zeta[self.q == 2]

        if len(H) > 0:
            return H[0] / 2

        return None

    def plot(self, figlabel_structure=None, figlabel_scaling=None):
        """
        Plots the structure functions.
        Args:
            fignum(int):  figure number; NOTE: fignum+1 can also be used to
                          plot the scaling function
        """

        if figlabel_structure is None:
            figlabel_structure = 'Structure Functions'

        if figlabel_scaling is None:
            figlabel_scaling = 'Scaling Function'

        if len(self.q) > 1:
            plot_dim_1 = 4
            plot_dim_2 = int(np.ceil(len(self.q) / 4.0))

        else:
            plot_dim_1 = 1
            plot_dim_2 = 1

        fig, axes = plt.subplots(plot_dim_1,
                                 plot_dim_2,
                                 num=figlabel_structure,
                                 squeeze=False,
                                 figsize=(30, 10))

        fig.suptitle(self.mrq_name +
                     r' - structure functions $\log_2(S(j,q))$')

        x = self.j
        for ind_q, q in enumerate(self.q):
            y = self.logvalues[ind_q, :]

            ax = axes[ind_q % 4][ind_q // 4]
            ax.plot(x, y, 'r--.')
            ax.set_xlabel('j')
            ax.set_ylabel(f'q = {q:.3f}')
            ax.grid()
            plt.draw()

            if len(self.zeta) > 0:
                # plot regression line
                x0 = self.j1
                x1 = self.j2
                slope = self.zeta[ind_q]
                intercept = self.intercept[ind_q]
                y0 = slope*x0 + intercept
                y1 = slope*x1 + intercept
                legend = 'slope = '+'%.5f' % (slope)

                ax.plot([x0, x1], [y0, y1], color='k',
                        linestyle='-', linewidth=2, label=legend)
                ax.legend()

        if len(self.q) > 1:
            plt.figure(figlabel_scaling)
            plt.plot(self.q, self.zeta, 'k--.')
            plt.xlabel('q')
            plt.ylabel(r'$\zeta(q)$')
            plt.suptitle(self.mrq_name + ' - scaling function')
            plt.grid()

        plt.draw()
