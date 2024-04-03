from dataclasses import dataclass, field, InitVar
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import MFractalBiVar
from ..regression import prepare_regression, prepare_weights

# For zeta(q1, q2) plot
# from mpl_toolkits.mplot3d import axes3d
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

from ..utils import fast_power
from ..regression import linear_regression
from ..multiresquantity import MultiResolutionQuantityBase, \
    MultiResolutionQuantity


@dataclass
class BiStructureFunction(MultiResolutionQuantityBase):
    mrq1: InitVar[MultiResolutionQuantity]
    mrq2: InitVar[MultiResolutionQuantity]
    q1: np.ndarray
    q2: np.ndarray
    scaling_ranges: List[Tuple[int]]
    bootstrapped_mfa: InitVar[MFractalBiVar] = None
    weighted: str = None
    j: np.ndarray = field(init=False)
    logvalues: np.ndarray = field(init=False)
    zeta: np.ndarray = field(init=False)
    intercept: np.ndarray = field(init=False)
    gamint: float = field(init=False)
    coherence: np.ndarray = field(init=False)

    def __post_init__(self, mrq1, mrq2, bootstrapped_mfa):

        self.n_rep = 1

        assert mrq1.formalism == mrq2.formalism
        self.formalism = mrq1.formalism

        assert mrq1.gamint == mrq2.gamint
        self.gamint = mrq1.gamint

        if any([(mrq1.nj[s] != mrq2.nj[s]).any() for s in mrq1.nj]):
            raise ValueError("Mismatch in number of coefficients between the "
                             "mrq")

        self.nj = mrq1.nj
        self.j = np.array(list(mrq1.values))

        if bootstrapped_mfa is not None:
            self.bootstrapped_obj = bootstrapped_mfa.structure

        self._compute(mrq1, mrq2)
        self._compute_zeta()

    def _compute(self, mrq1, mrq2):

        values = np.zeros((len(self.q1), len(self.q2), len(self.j)))
        abs_values = np.zeros(values.shape)

        for ind_j, j in enumerate(self.j):
            for ind_q1, q1 in enumerate(self.q1):
                for ind_q2, q2 in enumerate(self.q2):

                    values[ind_q1, ind_q2, ind_j] = \
                        np.nanmean(fast_power(mrq1.values[j], q1)
                                   * fast_power(mrq2.values[j], q2))

                    # s_j_q1_q2
                    abs_values[ind_q1, ind_q2, ind_j] = \
                        np.nanmean(fast_power(np.abs(mrq1.values[j]), q1)
                                   * fast_power(np.abs(mrq2.values[j]), q2))

        self.coherence = (values[self.q1 == 1, self.q2 == 1][0]
                          / np.sqrt((values[self.q1 == 2, self.q2 == 0])
                                    * (values[self.q1 == 0, self.q2 == 2])))[0]

        self.logvalues = np.log2(abs_values)

    def _compute_zeta(self):
        """
        Compute the value of the scale function zeta(q_1, q_2) for all q_1, q_2
        """
        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j)

        # x = x[:, :, :]  # No broadcasting repetitions for now

        # TODO adapt to new regression factorization

        n_j = self.logvalues.shape[2]

        y = self.logvalues.reshape(-1, n_j)[:, j_min_idx:j_max_idx, None, None]

        if self.weighted == 'bootstrap':

            # case where self is the bootstrapped mrq
            if self.bootstrapped_obj is None:
                std = self.STD_logvalues.reshape(-1, n_j)[
                    :, j_min_idx:j_max_idx]

            else:
                std = self.bootstrapped_obj.STD_logvalues.reshape(-1, n_j)[
                    :,
                    j_min - self.bootstrapped_obj.j.min():
                    j_max - self.bootstrapped_obj.j.min() + 1]

        else:
            std = None

        weights = prepare_weights(self, self.weighted, n_ranges, j_min, j_max,
                                  self.scaling_ranges, std)[:, :, :, 0, None]
        # No broadcasting repetitions for now

        nan_weighting = np.ones_like(y)
        nan_weighting[np.isnan(y)] = np.nan
        weights = weights * nan_weighting

        zeta, intercept = linear_regression(x, y, weights)

        n1 = self.logvalues.shape[0]
        n2 = self.logvalues.shape[1]

        self.zeta = zeta.reshape(n1, n2, n_ranges)
        self.intercept = intercept.reshape(n1, n2, n_ranges)

    def plot(self):

        fig, ax = plt.subplots(len(self.q1), len(self.q2), sharex=True,
                               figsize=(len(self.q1) * 5 + 2,
                                        len(self.q2) * 3 + 2))

        j_support = np.arange(self.j1, self.j2+1)

        slope_param = {
            'c': 'black',
            'linestyle': '--',
            'linewidth': 1.25
        }

        plot_param = {
            'linewidth': 2.5
        }

        for ind_q1, q1 in enumerate(self.q1):
            for ind_q2, q2 in enumerate(self.q2):

                ax[ind_q1, ind_q2].plot(self.j, self.logvalues[ind_q1, ind_q2],
                                        **plot_param)
                ax[ind_q1, ind_q2].plot(j_support,
                                        (j_support * self.zeta[ind_q1, ind_q2])
                                        + self.intercept[ind_q1, ind_q2],
                                        **slope_param,
                                        label=rf'$\zeta({q1}, {q2}): '
                                        f'{self.zeta[ind_q1, ind_q2]:.2f}$')
                ax[ind_q1, ind_q2].legend()

                ax[ind_q1, ind_q2].set_ylabel(rf'$S_{{{q1}, {q2}}}(j)$',
                                              size='large')
                ax[ind_q1, ind_q2].set_title(f'{q1=}, {q2=}', size='large')
                ax[-1, ind_q2].set_xlabel('j', size='large')

        sns.despine()

        return fig
