# from __future__ import print_function
# from __future__ import unicode_literals

from dataclasses import dataclass, field, InitVar

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    weighted: str = None
    j: np.array = field(init=False)
    m: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    log_cumulants: np.ndarray = field(init=False)
    slope: np.ndarray = field(init=False)
    intercept: np.ndarray = field(init=False)
    RHO_MF: np.ndarray = field(init=False)
    rho_mf: float = field(init=False)

    def __post_init__(self, mrq1, mrq2):

        self.nrep = 1

        assert mrq1.formalism == mrq2.formalism
        self.formalism = mrq1.formalism

        assert mrq1.nj == mrq2.nj
        self.nj = mrq1.nj
        self.j = np.array(list(mrq1.values))

        self.m = np.arange(0, self.n_cumul+1)

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

            for ind_m1, m1 in enumerate(self.m):
                for ind_m2, m2 in enumerate(self.m):

                    moments[ind_m1, ind_m2, ind_j] = \
                        np.nanmean((fast_power(log_T_X_j_1, m1))
                                   * fast_power(log_T_X_j_2, m2))

                    if m1 == m2 == 1:

                        self.values[ind_m1, ind_m2, ind_j] = \
                            (np.nanmean(fast_power(log_T_X_j_1, m1)
                                        * fast_power(log_T_X_j_2, m2))
                             - (self.values[ind_m1, 0, ind_j]
                                * self.values[0, ind_m2, ind_j]))

                    elif m1 + m2 == 1:
                        self.values[ind_m1, ind_m2, ind_j] = \
                            moments[ind_m1, ind_m2, ind_j]

                    elif (m1 == 0) ^ (m2 == 0):
                        aux = 0

                        if m1 == 0:
                            for ind_n, n in enumerate(np.arange(1, m2)):
                                aux += (binomial_coefficient(m2-1, n-1)
                                        * self.values[ind_m1, n, ind_j]
                                        * moments[ind_m1, ind_m2-ind_n-1,
                                                  ind_j])

                        elif m2 == 0:
                            for ind_n, n in enumerate(np.arange(1, m1)):
                                aux += (binomial_coefficient(m1-1, n-1)
                                        * self.values[n, ind_m2, ind_j]
                                        * moments[ind_m1-ind_n-1, ind_m2,
                                                  ind_j])

                        self.values[ind_m1, ind_m2, ind_j] = \
                            moments[ind_m1, ind_m2, ind_j] - aux

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

        if self.weighted:
            nj = self.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones((len(x), 1))

        ind_j1 = self.j1-1
        ind_j2 = self.j2-1

        for ind_m1, _ in enumerate(self.m):
            for ind_m2, _ in enumerate(self.m):

                y = self.values[ind_m1, ind_m2, ind_j1:ind_j2+1, None]
                slope, intercept = \
                    linear_regression(x, y, nj, return_variance=False)
                self.log_cumulants[ind_m1, ind_m2] = slope*log2_e
                self.slope[ind_m1, ind_m2] = slope
                self.intercept[ind_m1, ind_m2] = intercept

    def _compute_rho(self):

        if self.formalism == 'wavelet coef':
            self.RHO_MF = None
            self.rho_mf = None
        else:
            self.RHO_MF = (self.C11 / np.abs(np.sqrt(self.C02 * self.C20)))[0]
            self.rho_mf = -self.c11 / np.abs(np.sqrt(self.c02 * self.c20))

    def plot(self):

        fig_m, ax = plt.subplots(self.n_cumul, 2, sharex=True, figsize=(12, 7))

        j_support = np.arange(self.j1, self.j2 + 1)

        slope_param = {
            'c': 'black',
            'linestyle': '--',
            'linewidth': 1.25
        }

        plot_param = {
            'linewidth': 2.5
        }

        for i in range(self.n_cumul):

            ax[i, 0].plot(self.j, self.values[i+1, 0, :], **plot_param)
            ax[i, 0].set_ylabel(rf'$C{i+1}0$(j)', size='large')

            ax[i, 0].plot(j_support, ((j_support * self.slope[i+1, 0])
                                      + self.intercept[i+1, 0]),
                          **slope_param,
                          label=f'c{i+1}0: {self.log_cumulants[i+1, 0]:.2f}')
            ax[i, 0].legend()

            ax[i, 1].plot(self.j, self.values[0, i+1, :], **plot_param)
            ax[i, 1].set_ylabel(rf'$C0{i+1}$(j)', size='large')

            ax[i, 1].plot(j_support, ((j_support * self.slope[0, i+1])
                                      + self.intercept[0, i+1]),
                          **slope_param,
                          label=f'c0{i+1}: {self.log_cumulants[0, i+1]:.2f}')
            ax[i, 1].legend()

        ax[0, 0].set_title('X1', size='large')
        ax[0, 1].set_title('X2', size='large')
        ax[-1, 0].set_xlabel('j', size='large')

        sns.despine()

        fig_c = plt.figure()
        plt.plot(self.j, self.values[1, 1, :], **plot_param)
        plt.plot(j_support, ((j_support * self.slope[1, 1])
                             + self.intercept[1, 1]),
                 **slope_param,
                 label=f'c11: {self.slope[1, 1]:.2f}')
        plt.legend()
        plt.ylabel('$C11$(j)', size='large')
        plt.xlabel('j', size='large')

        sns.despine()

        return fig_m, fig_c

    def __getattr__(self, name):

        if name[0] == 'c' and len(name) == 3 and name[1:].isdigit():
            return self.log_cumulants[self.m == int(name[1]),
                                      self.m == int(name[2])]

        if name[0] == 'C' and len(name) == 3 and name[1:].isdigit():
            return self.values[self.m == int(name[1]),
                               self.m == int(name[2])]

        return self.__getattribute__(name)

    def compute_legendre(self, h_support=(0, 1.5), resolution=100):

        h_support = np.linspace(*h_support, resolution)

        b = (self.c20 * self.c02) - (self.c11 ** 2)

        L = np.ones((resolution, resolution))

        for i, h in enumerate(h_support):
            L[i, :] += self.c02 * b / 2 * (((h - self.c10) / b) ** 2)
            L[:, i] += self.c20 * b / 2 * (((h - self.c01) / b) ** 2)

        for i, h1 in enumerate(h_support):
            for j, h2 in enumerate(h_support):
                L[i, j] -= (self.c11 * b
                            * ((h1 - self.c10) / b)
                            * ((h2 - self.c01) / b))

        return h_support, L

    def plot_legendre(self, h_support=(0, 1.5), resolution=30,
                      figsize=(10, 10), cmap=None):

        h, L = self.compute_legendre(h_support=(0, 1.5),
                                     resolution=200)

        h_x = h[L.max(axis=0) >= 0]
        h_y = h[L.max(axis=1) >= 0]

        hmin = min([*h_x, *h_y])
        hmax = max([*h_x, *h_y])

        h, L = self.compute_legendre((hmin, hmax), resolution)

        cmap = cmap or plt.cm.coolwarm  # pylint: disable=no-member

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

        h_x = h[L.max(axis=0) >= 0]
        h_y = h[L.max(axis=1) >= 0]

        L = L[:, L.max(axis=0) >= 0]
        L = L[L.max(axis=1) >= 0, :]

        colors = cmap(L)
        colors[L < 0] = 0

        X, Y = np.meshgrid(h_x, h_y)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, L, cmap=cmap,
                               linewidth=1, antialiased=False,
                               vmin=0, vmax=1,
                               rstride=1, cstride=1,
                               facecolors=colors, shade=False, linestyle='-',
                               edgecolors='black', zorder=1)
        ax.set_zlim(0, 1)
        ax.view_init(elev=45)

        # TODO manage to plot the contours or switch to 3D plotting libs
        fig.colorbar(surf, shrink=0.6, aspect=10)

    def plot_legendre_pv(self, resolution=30, figsize=(10, 10), cmap=None,
                         use_ipyvtk=False):

        import pyvista as pv

        if use_ipyvtk:
            from ..viz import start_xvfb
            start_xvfb()

        h, L = self.compute_legendre(resolution=200)

        h_x = h[L.max(axis=0) >= 0]
        h_y = h[L.max(axis=1) >= 0]

        hmin = min([*h_x, *h_y])
        hmax = max([*h_x, *h_y])

        h, L = self.compute_legendre((hmin, hmax), resolution)

        # cmap = cmap or plt.cm.coolwarm  # pylint: disable=no-member

        h_x = h[L.max(axis=0) >= 0]
        h_y = h[L.max(axis=1) >= 0]

        L = L[:, L.max(axis=0) >= 0]
        L = L[L.max(axis=1) >= 0, :]

        X, Y = np.meshgrid(h_x, h_y)

        grid = pv.StructuredGrid(X, Y, L)
        bounds = [h_x.min(), h_x.max(), h_y.min(), h_y.max(), 0, 1]
        clipped = grid.clip_box(bounds, invert=False)

        p = pv.Plotter()
        p.add_mesh(clipped, scalars=clipped.points[:, 2])
        p.show_grid(xlabel='h1', ylabel='h2', zlabel='L(h1, h2)',
                    bounds=bounds)
        p.show(use_ipyvtk=use_ipyvtk)
