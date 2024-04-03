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

from ..utils import fast_power, isclose
from ..regression import linear_regression
from ..multiresquantity import WaveletDec
from ..scalingfunction import AbstractScalingFunction


@dataclass(kw_only=True)
class BiScalingFunction(AbstractScalingFunction):
    mrq1: InitVar[WaveletDec]
    mrq2: InitVar[WaveletDec]
    mode: str = 'all2all'
    gamint1: float = field(init=False)
    gamint2: float = field(init=False)
    n_sig: tuple[int] = field(init=False)

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

    def get_jrange(self, j1=None, j2=None, bootstrap=False):

        if self.bootstrapped_obj is not None and bootstrap:
            if j1 is None:
                j1 = self.bootstrapped_obj.j.min()
            if j2 is None:
                j2 = self.bootstrapped_obj.j.max()

        else:

            if j1 is None:
                j1 = self.j.min()
            if j2 is None:
                j2 = self.j.max()

        if self.j.min() > j1:
            raise ValueError(f"Expected mrq to have minium scale {j1=}, got "
                             f"{self.j.min()} instead")

        j_min = int(j1 - self.j.min())
        j_max = int(j2 - self.j.min() + 1)

        return j1, j2, j_min, j_max


@dataclass(kw_only=True)
class BiStructureFunction(BiScalingFunction):
    q1: np.ndarray
    q2: np.ndarray
    coherence: np.ndarray = field(init=False)

    def __post_init__(self, idx_reject, mrq1, mrq2):

        if mrq1.get_formalism() != mrq2.get_formalism():
            raise ValueError(
                'Multi-resolution quantities should have the same formalism, '
                f'currently is {mrq1.formalism=}, {mrq2.formalism=}')

        self.formalism = mrq1.get_formalism()
        self.gamint1 = mrq1.gamint
        self.gamint2 = mrq2.gamint

        # if mode == 'all2all':
        #     self.n_sig = mrq1.n_sig * mrq2.n_sig
        # elif mode == 'pairwise':
        #     if mrq1.n_sig != mrq2.n_sig:
        #         raise ValueError(
        #             'Pairwise mode needs equal number of signals on each '
        #             f'multi-resolution quantity, currently {mrq1.n_sig=} and '
        #             f'{mrq2.n_sig=}')
        #     self.n_sig = mrq1.n_sig

        # self.nj = mrq1.nj

        if max(mrq1.values) < max(mrq2.values):
            self.j = np.array(list(mrq1.values))
        else:
            self.j = np.array(list(mrq2.values))

        if self.bootstrapped_obj is not None:
            self.bootstrapped_obj = self.bootstrapped_obj.structure

        self._compute(mrq1, mrq2, idx_reject)
        self._compute_fit()

    def _compute(self, mrq1, mrq2, idx_reject):

        if ((ratio1 := mrq1.n_rep // mrq1.n_sig)
                != (ratio2 := mrq2.n_rep // mrq2.n_sig)):
            raise ValueError(
                'Mrq 1 and 2 have different number of bootstrapping '
                f'repetitions: {ratio1} and {ratio2}, respectively.')

        match self.mode:
            case 'all2all':
                n_rep = (mrq1.n_sig, mrq2.n_sig, ratio1)
            case 'pairwise':
                n_rep = (mrq1.n_sig, 1, ratio1)

        self.values = np.zeros(
            (len(self.q1), len(self.q2), len(self.j), len(self.scaling_ranges),
             *n_rep))
        self.coherence = np.zeros(self.values.shape[2:])

        for ind_j, j in enumerate(self.j):

            pow1 = {
                q1: fast_power(np.abs(
                    mrq1.get_values(j, idx_reject, True)), q1)[..., None, :]
                for q1 in self.q1 if q1 != 0
            }
            pow2 = {
                q2: fast_power(np.abs(
                    mrq2.get_values(j, idx_reject, True)), q2)
                for q2 in self.q2 if q2 != 0
            }

            match self.mode:
                case 'all2all':
                    pow2 = {k: v[..., None, :, :] for k, v in pow2.items()}
                case 'pairwise':
                    pow2 = {k: v[..., None, :] for k, v in pow2.items()}

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

            # Computing coherence
            val1 = mrq1.get_values(j, reshape=True)[..., None, :]

            match self.mode:
                case 'all2all':
                    val2 = mrq2.get_values(j, reshape=True)[..., None, :, :]
                case 'pairwise':
                    val2 = mrq2.get_values(j, reshape=True)[..., None, :]

            self.coherence[ind_j] = (
                np.nanmean(val1 * val2, axis=0)
                / np.sqrt(np.nanmean(fast_power(val1, 2) * fast_power(val2, 2),
                                     axis = 0)))
            
    def _compute_fit(self):
        """
        Compute the value of the scale function zeta(q_1, q_2) for all q_1, q_2
        """

        self.slope = np.zeros(
            (len(self.q1), len(self.q2), len(self.scaling_ranges),
             np.prod(self.values.shape[4:])))

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j)

        # x = x[:, :, :]  # No broadcasting repetitions for now

        # TODO adapt to new regression factorization

        nq1 = len(self.q1)
        nq2 = len(self.q2)

        y = self.values.reshape(
            nq1 * nq2, len(self.j), len(self.scaling_ranges),
            self.slope.shape[3])[:, j_min_idx:j_max_idx]

        if self.weighted == 'bootstrap':

            if self.bootstrapped_obj is None:
                std = self.std_values()[:, j_min_idx:j_max_idx]

            else:

                if j_min < self.bootstrapped_obj.j.min():
                    raise ValueError(
                        'Bootstrapped minimum scale '
                        f'{self.bootstrapped_obj.j.min()} inferior to minimum'
                        f'scale {j_min} used in estimation')
                
                std_slice = np.s_[
                    int(j_min - self.bootstrapped_obj.j.min()):
                    int(j_max - self.bootstrapped_obj.j.min() + 1)]
                
                std = self.bootstrapped_obj.std_values()[:, std_slice]

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

    def S_qq(self, q1, q2):
        return self.values[isclose(q1, self.q1), isclose(q2, self.q2)][0]

    def s_qq(self, q1, q2):

        out = self.slope[isclose(q1, self.q1), isclose(q2, self.q2)][0]
        out = out.reshape(len(self.scaling_ranges), *self.values.shape[4:])

        return out

    def plot(self, figsize=None, scaling_range=0, signal_idx1=0,
             signal_idx2=0, plot_CI=True, plot_scales=None, filename=None):

        if self.mode == 'pairwise':
            signal_idx2 = 0

        if plot_scales is None:
            j1, j2, j_min, j_max = self.get_jrange(None, None, plot_CI)
        else:
            j1, j2, j_min, j_max = self.get_jrange(*plot_scales, plot_CI)

        idx = np.s_[j_min:j_max]

        fig, axes = plt.subplots(len(self.q1), len(self.q2), sharex=True,
                                 figsize=figsize)

        x = self.j[idx]
        
        for ind_q1, q1 in enumerate(self.q1):
            for ind_q2, q2 in enumerate(self.q2):

                y = self.S_qq(q1, q2)[
                    idx, scaling_range, signal_idx1, signal_idx2, 0]

                if self.bootstrapped_obj is not None and plot_CI:

                    __, _, j_min_CI, j_max_CI = self.bootstrapped_obj.get_jrange(
                    j1, j2)

                    CI = self.CIE_S_qq(q1, q2)[
                        j_min_CI:j_max_CI, scaling_range, signal_idx1,
                        signal_idx2]

                    CI -= y[:, None]
                    CI[:, 1] *= -1
                    assert (CI < 0).sum() == 0
                    CI = CI.transpose()

                else:
                    CI = None

                ax = axes[ind_q1, ind_q2]
                ax.errorbar(x, y, CI)
                # ax.tick_params(bottom=False, top=False, which='minor')
                ax.set(xlabel='Temporal scale $j$',
                       ylabel=f'$q_1={q1:.1f}$, $q_2={q2:.1f}$')
                
                x0, x1 = self.scaling_ranges[scaling_range]

                idx_a = np.s_[ind_q1, ind_q2, scaling_range]
                idx_b = np.s_[signal_idx1, signal_idx2, 0]

                slope = self.slope[idx_a].reshape(self.values.shape[4:])[idx_b]
                intercept = self.intercept[idx_a].reshape(
                    self.values.shape[4:])[idx_b]
                
                assert x0 in x, "Scaling range not included in plotting range"
                assert x1 in x, "Scaling range not included in plotting range"

                y0 = slope*x0 + intercept
                y1 = slope*x1 + intercept

                if self.bootstrapped_obj is not None and plot_CI:
                    CI = self.CIE_s_qq(q1, q2)[scaling_range]
                    CI_legend = f"; [{CI[0]:.1f}, {CI[1]:.1f}]"
                else:
                    CI_legend = ""

                legend = rf'$s_{{{q1:d}{q2:d}}}$ = {slope:.2f}' + CI_legend

                ax.plot([x0, x1], [y0, y1], color='k',
                    linestyle='-', linewidth=2, label=legend, zorder=5)
                ax.legend()

        if filename is not None:
            plt.savefig(filename)


@dataclass(kw_only=True)
class BiCumulants(BiScalingFunction):
    cm: int = field(init=False)
