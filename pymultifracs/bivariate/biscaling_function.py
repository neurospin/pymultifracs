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
    scaling_ranges: list[tuple[int]]
    mode: InitVar[str] = 'all2all'
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

        if mrq1.get_formalism() != mrq2.get_formalism():
            raise ValueError(
                'Multi-resolution quantities should have the same formalism, '
                f'currently is {mrq1.formalism=}, {mrq2.formalism=}')

        self.formalism = mrq1.get_formalism()
        self.gamint1 = mrq1.gamint
        self.gamint2 = mrq2.gamint

        if ((ratio1 := mrq1.n_rep // mrq1.n_sig)
                != (ratio2 := mrq2.n_rep // mrq2.n_sig)):
            raise ValueError(
                'Mrq 1 and 2 have different number of bootstrapping '
                f'repetitions: {ratio1} and {ratio2}, respectively.')

        if mode == 'all2all':
            self.n_sig = mrq1.n_sig * mrq2.n_sig
        elif mode == 'pairwise':
            if mrq1.n_sig != mrq2.n_sig:
                raise ValueError(
                    'Pairwise mode needs equal number of signals on each '
                    f'multi-resolution quantity, currently {mrq1.n_sig=} and '
                    f'{mrq2.n_sig=}')
            self.n_sig = mrq1.n_sig

        n_rep = ratio1 * self.n_sig

        # self.nj = mrq1.nj
        self.j = np.array(list(mrq1.values))

        if self.bootstrapped_sf is not None:
            self.bootstrapped_mrq = self.bootstrapped_sf.structure

        self._compute(mrq1, mrq2)
        self._compute_fit()

    def _compute(self, mrq1, mrq2):

        self.values = np.zeros(
            (len(self.q1), len(self.q2), len(self.j), len(self.scaling_ranges), mrq1.n_rep, mrq2.n_rep))
        self.coherence = np.zeros(self.values.shape[2:])

        for ind_j, j in enumerate(self.j):

            pow1 = {
                q1: fast_power(np.abs(mrq1.get_values(j)), q1)[..., None]
                for q1 in self.q1 if q1 != 0
            }
            pow2 = {
                q2: fast_power(np.abs(mrq2.get_values(j)), q2)[..., None, :]
                for q2 in self.q2 if q2 != 0
            }

            for ind_q1, q1 in enumerate(self.q1):
                for ind_q2, q2 in enumerate(self.q2):

                    if q1 == q2 == 0:
                        self.values[ind_q1, ind_q2, ind_j] = 0
                        continue

                    if q1 == 0:
                        self.values[ind_q1, ind_q2, ind_j] = np.log2(
                            np.nanmean(pow2[q2], axis=0))
                        continue

                    if q2 == 0:
                        self.values[ind_q1, ind_q2, ind_j] = np.log2(
                            np.nanmean(pow1[q1], axis=0))
                        continue

                    self.values[ind_q1, ind_q2, ind_j] = np.log2(
                        np.nanmean(pow1[q1] * pow2[q2], axis=0))

            self.coherence[ind_j] = (
                np.nanmean(mrq1.values[j][..., None]
                           * mrq2.values[j][..., None, :],
                           axis=0)
                / np.sqrt(np.nanmean(
                    fast_power(mrq1.values[j], 2)[..., None]
                    * fast_power(mrq2.values[j], 2)[..., None, :],
                    axis = 0)))
            
    def _compute_fit(self):
        """
        Compute the value of the scale function zeta(q_1, q_2) for all q_1, q_2
        """

        self.slope = np.zeros(
            (len(self.q1), len(self.q2), len(self.scaling_ranges),
             self.values.shape[-1]))

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j)

        # x = x[:, :, :]  # No broadcasting repetitions for now

        # TODO adapt to new regression factorization

        nq1 = len(self.q1)
        nq2 = len(self.q2)

        N1, N2 = self.values.shape[-2:]

        y = self.values.reshape(nq1 * nq2, len(self.j), N1 * N2)[
            :, j_min_idx:j_max_idx]

        if self.weighted == 'bootstrap':

            if self.bootstrapped_sf is None:
                std = self.std_values()[:, j_min_idx:j_max_idx]

            else:

                if j_min < self.bootstrapped_sf.j.min():
                    raise ValueError(
                        'Bootstrapped minimum scale '
                        f'{self.bootstrapped_sf.j.min()} inferior to minimum'
                        f'scale {j_min} used in estimation')
                
                std_slice = np.s_[
                    int(j_min - self.bootstrapped_sf.j.min()):
                    int(j_max - self.bootstrapped_sf.j.min() + 1)]
                
                std = self.bootstrapped_sf.std_values()[:, std_slice]

        else:
            std = None

        self.weights = prepare_weights(self, self.weighted, n_ranges, j_min,
                                       j_max, self.scaling_ranges, y, std)

        # nan_weighting = np.ones_like(y)
        # nan_weighting[np.isnan(y)] = np.nan

        # weights = weights * nan_weighting

        self.slope, self.intercept = linear_regression(x, y, self.weights)

        n1 = self.values.shape[0]
        n2 = self.values.shape[1]

        self.slope = self.slope.reshape(n1, n2, n_ranges, -1)
        self.intercept = self.intercept.reshape(n1, n2, n_ranges, -1)

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


@dataclass(kw_only=True)
class BiCumulants(BiScalingFunction):
    cm: int = field(init=False)
