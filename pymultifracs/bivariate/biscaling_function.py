from typing import Any
from dataclasses import dataclass, field, InitVar
import inspect

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
from ..multiresquantity import WaveletDec

@dataclass
class BiScalingFunction:
    mrq1: InitVar[WaveletDec]
    mrq2: InitVar[WaveletDec]
    mode: InitVar[str] = 'all2all'
    scaling_ranges: list[tuple[int]]
    bootstrapped_sf: Any | None = None
    formalism: str = field(init=False)
    gamint1: float = field(init=False)
    gamint2: float = field(init=False)
    n_sig: int = field(init=False)
    j: np.ndarray = field(init=False)

    @classmethod
    def from_dict(cls, d):
        r"""Method to instanciate a dataclass by passing a dictionary with
        extra keywords

        Parameters
        ----------
        d : dict
            Dictionary containing at least all the parameters required by
            __init__, but can also contain other parameters, which will be
            ignored

        Returns
        -------
        MultiResolutionQuantityBase
            Properly initialized multi resolution quantity

        Notes
        -----
        .. note:: Normally, dataclasses can only be instantiated by only
                  specifiying parameters expected by the automatically
                  generated __init__ method.
                  Using this method instead allows us to discard extraneous
                  parameters, similarly to introducing a \*\*kwargs parameter.
        """
        return cls(**{
            k: v for k, v in d.items()
            if k in inspect.signature(cls).parameters
        })
    
    def _check_enough_rep_bootstrap(self):
        if (ratio := self.n_rep // self.n_sig) < 2:
            raise ValueError(
                f'n_rep = {ratio} per original signal too small to build '
                'confidence intervals'
                )


@dataclass(kw_only=True)
class BiStructureFunction(BiScalingFunction):
    q1: np.ndarray
    q2: np.ndarray
    weighted: str = None
    j: np.ndarray = field(init=False)
    logvalues: np.ndarray = field(init=False)
    slope: np.ndarray = field(init=False)
    intercept: np.ndarray = field(init=False)
    gamint: float = field(init=False)
    coherence: np.ndarray = field(init=False)

    def __post_init__(self, mrq1, mrq2, mode):

        if mrq1.formalism != mrq2.formalism:
            raise ValueError(
                'Multi-resolution quantities should have the same formalism, '
                f'currently is {mrq1.formalism=}, {mrq2.formalism=}')

        self.formalism = mrq1.formalism
        self.gamint1 = mrq1.gamint
        self.gamint2 = mrq2.gamint

        if mode == 'all2all':
            self.n_sig = mrq1.n_sig * mrq2.n_sig
        elif mode == 'pairwise':
            if mrq1.n_sig != mrq2.n_sig:
                raise ValueError(
                    'Pairwise mode needs equal number of signals on each multi-resolution quantity, '
                    f'currently {mrq1.n_sig=} and {mrq2.n_sig=}')
            self.n_sig = mrq1.n_sig

        # self.n_rep = 1

        # if any([(mrq1.nj[s] != mrq2.nj[s]).any() for s in mrq1.nj]):
        #     raise ValueError("Mismatch in number of coefficients between the "
        #                      "mrq")

        # self.nj = mrq1.nj
        self.j = np.array(list(mrq1.values))

        if self.bootstrapped_sf is not None:
            self.bootstrapped_mrq = self.bootstrapped_sf.structure

        self._compute(mrq1, mrq2)
        self._compute_zeta()

    def _compute(self, mrq1, mrq2):

        self.values = np.zeros((len(self.q1), len(self.q2), len(self.j), ))
        self.coherence = np.zeros(self.values[2:])

        pow1 = {
            q1: fast_power(np.abs(mrq1.values[j]), q1)
            for q1 in self.q1 if q2 != 0
        }
        pow2 = {
            q2: fast_power(np.abs(mrq2.values[j]), q2)
            for q2 in self.q2 if q2 != 0
        }

        for ind_j, j in enumerate(self.j):
            for ind_q1, q1 in enumerate(self.q1):
                for ind_q2, q2 in enumerate(self.q2):

                    if q1 == 0:
                        self.values[ind_q1, ind_q2, ind_j] = np.log2(np.nanmean(pow2[q2], axis=0))
                        continue

                    if q2 == 0:
                        self.values[ind_q1, ind_q2, ind_j] = np.log2(np.nanmean(pow1[q1], axis=0))
                        continue

                    self.values[ind_q1, ind_q2, ind_j] = np.log2(np.nanmean(pow1[q1] * pow2[q2], axis=0))

            self.coherence[j] = mrq1.values[j] * mrq2.values[j] / np.sqrt(
            np.nanmean(fast_power(mrq1.values[j], 2)) * fast_power(mrq2.values[j], 2)
        )

    def _compute_zeta(self):
        """
        Compute the value of the scale function zeta(q_1, q_2) for all q_1, q_2
        """

        self.slope = np.zeros((len(self.q1), len(self.q2), len(self.scaling_ranges), self.values.shape[-1]))

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j)

        # x = x[:, :, :]  # No broadcasting repetitions for now

        # TODO adapt to new regression factorization

        y = self.values.reshape(-1, *self.slope.shape[2:])

        y = self.logvalues.reshape(-1, n_j)[:, j_min_idx:j_max_idx, None, None]

        if self.weighted == 'bootstrap':

            # case where self is the bootstrapped mrq
            if self.bootstrapped_mrq is None:
                std = self.STD_logvalues.reshape(-1, n_j)[
                    :, j_min_idx:j_max_idx]

            else:
                std = self.bootstrapped_mrq.STD_logvalues.reshape(-1, n_j)[
                    :,
                    j_min - self.bootstrapped_mrq.j.min():
                    j_max - self.bootstrapped_mrq.j.min() + 1]

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
