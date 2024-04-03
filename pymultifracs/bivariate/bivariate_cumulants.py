# from __future__ import print_function
# from __future__ import unicode_literals

from dataclasses import dataclass, field, InitVar
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as binomial_coefficient

from ..regression import linear_regression, prepare_regression, prepare_weights
from ..utils import MFractalVar, fast_power
from ..multiresquantity import MultiResolutionQuantity, \
    MultiResolutionQuantityBase
from ..viz import plot_bicm


@dataclass
class BiCumulants(MultiResolutionQuantityBase):
    mrq1: InitVar[MultiResolutionQuantity]
    mrq2: InitVar[MultiResolutionQuantity]
    n_cumul: int
    scaling_ranges: List[Tuple[int]]
    weighted: str = None
    bootstrapped_mfa: InitVar[MFractalVar] = None
    j: np.array = field(init=False)
    m: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    log_cumulants: np.ndarray = field(init=False)
    slope: np.ndarray = field(init=False)
    intercept: np.ndarray = field(init=False)
    RHO_MF: np.ndarray = field(init=False)
    rho_mf: float = field(init=False)

    def __post_init__(self, mrq1, mrq2, bootstrapped_mfa):

        # self.n_rep = 1
        self.n_sig = 1

        assert mrq1.formalism == mrq2.formalism
        self.formalism = mrq1.formalism

        if bootstrapped_mfa is not None:
            self.bootstrapped_obj = bootstrapped_mfa.cumulants

        if any([(mrq1.nj[s] != mrq2.nj[s]).any() for s in mrq1.nj]):
            raise ValueError("Mismatch in number of coefficients between the "
                             "mrq")

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

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j
        )

        # self.log_cumulants = np.zeros(self.values.shape[:2])
        # self.slope = np.zeros(self.log_cumulants.shape)
        # self.intercept = np.zeros(self.log_cumulants.shape)

        log2_e = np.log2(np.exp(1))
        # x = np.arange(self.j1, self.j2+1)[:, None]

        n_j = self.values.shape[2]

        y = self.values.reshape(-1, n_j)[:, j_min_idx:j_max_idx, None, None]

        if self.weighted == 'bootstrap':

            # case where self is the bootstrapped mrq
            if self.bootstrapped_cm is None:
                std = self.STD_values.reshape(-1, n_j)[:, j_min_idx:j_max_idx]

            else:
                std = self.bootstrapped_cm.STD_values.reshape(-1, n_j)[
                    :,
                    j_min - self.bootstrapped_cm.j.min():
                    j_max - self.bootstrapped_cm.j.min() + 1]

        else:
            std = None

        weights = prepare_weights(self, self.weighted, n_ranges, j_min, j_max,
                                  self.scaling_ranges, std)[:, :, :, 0, None]
        # No broadcasting repetitions for now

        nan_weighting = np.ones_like(y)
        nan_weighting[np.isnan(y)] = np.nan
        weights = weights * nan_weighting

        self.slope, self.intercept = linear_regression(x, y, weights)

        n1 = self.values.shape[0]
        n2 = self.values.shape[1]

        self.slope = self.slope.reshape(n1, n2, n_ranges)
        self.intercept = self.intercept.reshape(n1, n2, n_ranges)

        self.log_cumulants = log2_e * self.slope

    def _compute_rho(self):

        if self.formalism == 'wavelet coef':
            self.RHO_MF = None
            self.rho_mf = None
        else:
            self.RHO_MF = (self.C11 / np.abs(np.sqrt(self.C02 * self.C20)))
            self.rho_mf = -self.c11 / np.abs(np.sqrt(self.c02 * self.c20))

    def plot(self, figsize=(6, 4), j1=None, scaling_range=0, filename=None):

        if j1 is None:
            j1 = self.j.min()

        if self.j.min() > j1:
            raise ValueError(f"Expected mrq to have minium scale {j1=}, got "
                             f"{self.j.min()} instead")

        ncol = len(self.m)

        fig, axes = plt.subplots(ncol,
                                 ncol,
                                 squeeze=False,
                                 figsize=figsize,
                                 sharex=True)

        fig.suptitle(self.formalism + r" - bivariate cumulants $C_{m, m'}(j)$")

        for ind_m1, m1 in enumerate(self.m):
            for ind_m2, m2 in enumerate(self.m):

                plot_bicm(self, ind_m1, ind_m2, j1, None, scaling_range,
                          axes[ind_m1][ind_m2], plot_legend=True)

        # for j in range(ind_m1):
        #     axes[j % ncol][j // ncol].xaxis.set_visible(False)

        # for j in range(ind_m1 + 1, len(axes.flat)):
        #     fig.delaxes(axes[j % ncol][j // ncol])

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename)

    def __getattr__(self, name):

        if name[0] == 'c' and len(name) == 3 and name[1:].isdigit():
            return self.log_cumulants[self.m == int(name[1]),
                                      self.m == int(name[2])][0]

        if name[0] == 'C' and len(name) == 3 and name[1:].isdigit():
            return self.values[self.m == int(name[1]),
                               self.m == int(name[2])][0]

        # if name == 'n_rep':
        #     return self.log_cumulants.shape[-1]
        #     return 1

        if (super_attr := super().__getattr__(name)) is not None:
            return super_attr

        return super().__getattribute__(name)

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