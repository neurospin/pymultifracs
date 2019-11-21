from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from .utils import Utils

# For zeta(q1, q2) plot
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class BiStructureFunction:
    """
    This class provides methods for computing and analyzing bivariate struture
    functions S(j, q1, q2)

    Args:
        mrq_1 (MultiResolutionQuantity):   multiresolution quantity of signal 1

        mrq_2 (MultiResolutionQuantity):   multiresolution quantity of signal 2


        q_1 (numpy.array)             :  list of exponents q1

        q_2 (numpy.array)             :  list of exponents q2

        j1 (int)                     : smallest scale analysis

        j2 (int)                     : largest scale analysis

        wtype (int)                  : 0 for ordinary regression
                                       1 for weighted regression

        values (numpy.array): values[ind_q1, ind_q2, ind_j] =
                               values of S(j, q1, q2),
                              with q_i = self.q_i[ind_qi] and j = self.j[ind_j]

        logvalues (numpy.array): logvalues[ind_q1, ind_q2, ind_j] =
                                  values of log_2 (S(j, q1, q2)),
                                 with q_i = self.q_i[ind_qi]
                                  and j = self.j[ind_j]

    """
    def __init__(self, mrq_1, mrq_2, q_1, q_2, j1, j2, wtype):
        self.mrq_1 = mrq_1
        self.mrq_2 = mrq_2
        self.q_1 = q_1
        self.q_2 = q_2
        self.j1 = j1
        self.j2 = j2
        self.j = np.array(list(mrq_1.values))
        self.wtype = wtype
        self.values = np.zeros((len(self.q_1), len(self.q_2), len(self.j)))
        self.logvalues = np.zeros((len(self.q_1), len(self.q_2), len(self.j)))
        self.zeta = []
        self.utils = Utils()  # used for linear regression
        self._compute()
        self._compute_zeta()

    # j should be the same as np.array(list(mrq_2.values))

    def _compute(self):
        for ind_q1, q1 in enumerate(self.q_1):
            for ind_q2, q2 in enumerate(self.q_2):
                for ind_j, j in enumerate(self.j):
                    c_j_1 = np.abs(self.mrq_1.values[j])
                    c_j_2 = np.abs(self.mrq_2.values[j])

                    s_j_q1_q2 = np.mean((c_j_1**q1)*(c_j_2**q2))
                    self.values[ind_q1, ind_q2, ind_j] = s_j_q1_q2

        self.logvalues = np.log2(self.values)

    def _compute_zeta(self):
        """
        Compute the value of the scale function zeta(q_1, q_2) for all q_1, q_2
        """
        self.zeta = np.zeros((len(self.q_1), len(self.q_2)))
        self.intercept = np.zeros((len(self.q_1), len(self.q_2)))

        x = np.arange(self.j1, self.j2+1)

        if self.wtype == 1:
            nj = self.mrq_1.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones(len(x))

        ind_j1 = self.j1-1
        ind_j2 = self.j2-1
        for ind_q1, q1 in enumerate(self.q_1):
            for ind_q2, q2 in enumerate(self.q_2):
                y = self.logvalues[ind_q1, ind_q2, ind_j1:ind_j2+1]
                slope, intercept = self.utils.linear_regression(x, y, nj)
                self.zeta[ind_q1, ind_q2] = slope
                self.intercept[ind_q1, ind_q2] = intercept

    def plot(self, figlabel_structure=None, figlabel_scaling=None):
        """
        Plots the structure functions.
        Args:
            fignum(int):  figure number; NOTE: fignum+1 can also be used to
                                               plot the scaling function
        """

        if figlabel_structure is None:
            figlabel_structure = 'Bivariate Structure Functions'

        if figlabel_scaling is None:
            figlabel_scaling = 'Scaling Function'

        nq = len(self.q_1)*len(self.q_2)

        if nq > 1:
            plot_dim_1 = 5
            plot_dim_2 = int(np.ceil(nq / 5.0))

        else:
            plot_dim_1 = 1
            plot_dim_2 = 1

        fig, axes = plt.subplots(plot_dim_1,
                                 plot_dim_2,
                                 num=figlabel_structure,
                                 squeeze=False, sharex=True)

        fig.suptitle(self.mrq_1.name +
                     ' - bivariate structure functions ' +
                     r'$\log_2(S(j,q_1, q_2))$')

        x = self.j

        plot_index = -1
        for ind_q1, q1 in enumerate(self.q_1):
            for ind_q2, q2 in enumerate(self.q_2):

                plot_index += 1

                y = self.logvalues[ind_q1, ind_q2, :]

                ax = axes[plot_index % 5][plot_index // 5]
                ax.plot(x, y, 'r--.')
                ax.set_xlabel('j')
                ax.set_title(f'$q_1$ = {q1:0.1f},  $q_2$ = {q2:0.1f}')
                ax.grid()
                plt.draw()

                if len(self.zeta) > 0:
                    # plot regression line
                    x0 = self.j1
                    x1 = self.j2
                    slope = self.zeta[ind_q1, ind_q2]
                    intercept = self.intercept[ind_q1, ind_q2]
                    y0 = slope*x0 + intercept
                    y1 = slope*x1 + intercept
                    legend = 'slope = '+'%.5f' % (slope)

                    ax.plot([x0, x1], [y0, y1], color='k',
                            linestyle='-', linewidth=2, label=legend)
                    ax.legend()

        if nq > 1:
            fig = plt.figure(figlabel_scaling)
            ax = fig.add_subplot(111, projection='3d')
            Q1, Q2 = np.meshgrid(self.q_1, self.q_2)
            surf = ax.plot_surface(Q1, Q2, self.zeta, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            ax.set_xlabel('$q_1$')
            ax.set_ylabel('$q_2$')
            ax.set_zlabel(r'$\zeta(q_1, q_2)$')
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.suptitle(self.mrq_1.name + ' - bivariate scaling function')
            plt.grid()

            plt.draw()
