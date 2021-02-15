from dataclasses import dataclass, field, InitVar

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For zeta(q1, q2) plot
# from mpl_toolkits.mplot3d import axes3d
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

from ..utils import fast_power, linear_regression
from ..multiresquantity import MultiResolutionQuantityBase, \
    MultiResolutionQuantity


@dataclass
class BiStructureFunction(MultiResolutionQuantityBase):
    mrq1: InitVar[MultiResolutionQuantity]
    mrq2: InitVar[MultiResolutionQuantity]
    j1: int
    j2: int
    q1: np.ndarray
    q2: np.ndarray
    weighted: bool
    j: np.ndarray = field(init=False)
    logvalues: np.ndarray = field(init=False)
    zeta: np.ndarray = field(init=False)
    intercept: np.ndarray = field(init=False)
    H: float = field(init=False)
    gamint: float = field(init=False)
    coherence: np.ndarray = field(init=False)

    def __post_init__(self, mrq1, mrq2):

        self.nrep = 1

        assert mrq1.formalism == mrq2.formalism
        self.formalism = mrq1.formalism

        assert mrq1.gamint == mrq2.gamint
        self.gamint = mrq1.gamint

        assert mrq1.nj == mrq2.nj
        self.nj = mrq1.nj
        self.j = np.array(list(mrq1.values))

        self._compute(mrq1, mrq2)
        self._compute_zeta()

    def _compute(self, mrq1, mrq2):

        values = np.zeros((len(self.q1), len(self.q2), len(self.j)))
        abs_values = np.zeros(values.shape)

        for ind_q1, q1 in enumerate(self.q1):
            for ind_q2, q2 in enumerate(self.q2):
                for ind_j, j in enumerate(self.j):

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
        self.zeta = np.zeros(self.logvalues.shape[:2])
        self.intercept = np.zeros(self.zeta.shape)

        x = np.arange(self.j1, self.j2+1)[:, None]

        if self.weighted == 1:
            nj = self.get_nj_interv(self.j1, self.j2)
        else:
            nj = np.ones((len(x), 1))

        ind_j1 = self.j1-1
        ind_j2 = self.j2-1

        for ind_q1, _ in enumerate(self.q1):
            for ind_q2, _ in enumerate(self.q2):

                y = self.logvalues[ind_q1, ind_q2, ind_j1:ind_j2+1, None]
                slope, intercept = linear_regression(x, y, nj)
                self.zeta[ind_q1, ind_q2] = slope
                self.intercept[ind_q1, ind_q2] = intercept

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
