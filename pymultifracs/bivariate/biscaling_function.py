"""
Authors: Merlin Dumeur <merlin@dumeur.net>
         Omar D. Domingues <omar.darwiche-domingues@inria.fr>
"""

from dataclasses import dataclass, field, InitVar

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import special


from ..regression import prepare_regression, prepare_weights

# For zeta(q1, q2) plot
# from mpl_toolkits.mplot3d import axes3d
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

from ..utils import fast_power, isclose, Dim
from ..viz import plot_bicm
from ..regression import linear_regression
from ..multiresquantity import WaveletDec
from ..scalingfunction import AbstractScalingFunction


@dataclass(kw_only=True)
class BiScalingFunction(AbstractScalingFunction):
    """
    Bivariate general scaling function
    """
    mrq1: InitVar[WaveletDec]
    mrq2: InitVar[WaveletDec]
    min_j: InitVar[int] = 1
    mode: str = 'all2all'
    gamint1: float = field(init=False)
    gamint2: float = field(init=False)
    n_channel: tuple[int] = field(init=False)
    nj_margin: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self, idx_reject, mrq1, mrq2, min_j):

        if mrq1.get_formalism() != mrq2.get_formalism():
            raise ValueError(
                'Multi-resolution quantities should have the same formalism, '
                f'currently is {mrq1.formalism=}, {mrq2.formalism=}')

        self.formalism = mrq1.get_formalism()
        self.gamint1 = mrq1.gamint
        self.gamint2 = mrq2.gamint

        match self.mode:
            case 'all2all':
                self.n_channel = (mrq1.n_channel, mrq2.n_channel)
            case 'pairwise':
                if mrq1.n_channel != mrq2.n_channel:
                    raise ValueError(
                        'Pairwise mode needs equal number of signals on each '
                        f'multi-resolution quantity, currently {mrq1.n_channel=} '
                        f'and {mrq2.n_channel=}')
                self.n_channel = (mrq1.n_channel, 1)

        if max(mrq1.values) < max(mrq2.values):
            self.j = np.array(list(mrq1.values))
        else:
            self.j = np.array(list(mrq2.values))

        self.j = self.j[self.j >= min_j]

        self.nj_margin = np.array([mrq1.get_nj_interv(min_j), mrq2.get_nj_interv(min_j)])
        self.nj = (
            (self.nj_margin[0][:, None] + self.nj_margin[1][..., None]) / 2
            ).reshape(len(self.j), len(self.scaling_ranges), -1)

    def get_nj_interv(self, j_min, j_max):
        """
        Get number of coefficients per scale.
        """
        return self.nj[j_min-min(self.j):j_max-min(self.j)+1]

    def get_nj_interv_margin(self, j_min, j_max, margin):
        """
        Get number of coefficients per scale for the margins.
        """
        return self.nj_margin[margin, j_min-min(self.j):j_max-min(self.j)+1]

    def _check_enough_rep_bootstrap(self):
        if (ratio := self.n_rep // self.n_channel) < 2:
            raise ValueError(
                f'n_rep = {ratio} per original signal too small to build '
                'confidence intervals'
                )

    def get_jrange(self, j1=None, j2=None, bootstrap=False):
        """
        Get the range of j values.
        """

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

    def _prepare_nrep(self, mrq1, mrq2):

        match self.mode:
            case 'all2all':
                n_rep = (mrq1.n_channel, mrq2.n_channel)
            case 'pairwise':
                n_rep = (mrq1.n_channel, 1)

        flag_bootstrap1 = 'bootstrap' in mrq1.dims
        flag_bootstrap2 = 'bootstrap' in mrq2.dims

        if (
                (flag_bootstrap1 and not flag_bootstrap2)
                or (flag_bootstrap2 and not flag_bootstrap1)):

            raise ValueError(
                'One of the Mrqs has been bootstrapped but not the other.')

        if flag_bootstrap1 and flag_bootstrap2:

            if ((ratio1 := mrq1.n_rep // mrq1.n_channel)
                    != (ratio2 := mrq2.n_rep // mrq2.n_channel)):
                raise ValueError(
                    'Mrq 1 and 2 have different number of bootstrapping '
                    f'repetitions: {ratio1} and {ratio2}, respectively.')

            n_rep.append(ratio1)

        return n_rep

    def _compute_fit(self, mrq1, mrq2, margin=None, value_name=None):
        """
        Compute the value of the scale function zeta(q_1, q_2) for all q_1, q_2
        """

        if value_name is not None:
            values = getattr(self, value_name)
        else:
            values = self.values

        # n1, n2 = values.shape[:2]

        # self.slope = np.zeros(
        #     (n1, n2, len(self.scaling_ranges), np.prod(values.shape[4:])))

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j, values.dims)

        # y = values.reshape(
            # n1 * n2, len(self.j), len(self.scaling_ranges),
            # np.prod(values.shape[4:]))[:, j_min_idx:j_max_idx]

        y = values.sel(j=slice(j_min, j_max))

        if self.weighted == 'bootstrap':

            if self.bootstrapped_obj is None:
                std = self.std_values().sel(j=slice(j_min,j_max))

            else:

                if j_min < self.bootstrapped_obj.j.min():
                    raise ValueError(
                        'Bootstrapped minimum scale '
                        f'{self.bootstrapped_obj.j.min()} inferior to minimum'
                        f'scale {j_min} used in estimation')

                # std_slice = np.s_[
                #     int(j_min - self.bootstrapped_obj.j.min()):
                #     int(j_max - self.bootstrapped_obj.j.min() + 1)]

                std = self.bootstrapped_obj.std_values().sel(j=slice(j_min, j_max))

        else:
            std = None

        if margin is None:
            nj_fun = self.get_nj_interv
        else:
            nj_fun = lambda x, y: self.get_nj_interv_margin(x, y, margin)

        weights = prepare_weights(nj_fun, self.weighted, n_ranges, j_min,
                                  j_max, self.scaling_ranges, y, std)

        slope, intercept = linear_regression(x, y, weights)

        # slope = slope.reshape(n1, n2, n_ranges, *values.shape[4:])
        # intercept = intercept.reshape(n1, n2, n_ranges, *values.shape[4:])

        return slope, intercept, weights

        # if out_prefix is not None:
        #     setattr(self, out_prefix + '_slope', slope)
        #     setattr(self, out_prefix + '_intercept', intercept)
        #     setattr(self, out_prefix + '_weights', weights)
        # else:
        #     self.slope = slope
        #     self.intercept = intercept
        #     self.weights = weights


@dataclass(kw_only=True)
class BiStructureFunction(BiScalingFunction):
    """
    Bivariate structure function.
    """
    q1: np.ndarray
    q2: np.ndarray
    coherence: np.ndarray = field(init=False)

    def __post_init__(self, idx_reject, mrq1, mrq2, min_j):

        super().__post_init__(idx_reject, mrq1, mrq2, min_j)

        if self.bootstrapped_obj is not None:
            self.bootstrapped_obj = self.bootstrapped_obj.structure

        self._compute(mrq1, mrq2, idx_reject)
        self.slope, self.intercept, self.weights = self._compute_fit(
            mrq1, mrq2)

    def _compute(self, mrq1, mrq2, idx_reject):

        # 1 - Define the shape of the values and coherence arrays
        shape = (
            len(self.q1), len(self.q2), len(self.j), len(self.scaling_ranges),
        )
        dims = (Dim.q1, Dim.q2, Dim.j, Dim.scaling_range)

        mrq_sizes = mrq1.get_values(self.j.max()).sizes

        match self.mode:

            case 'all2all':

                mrq_dims = [d for d in mrq_sizes
                            if d not in [Dim.channel, Dim.k_j, *dims]]
                mrq_shapes = [s for d, s in mrq_sizes.items()
                              if d not in [Dim.channel, Dim.k_j, *dims]]

                dims = (*dims, *mrq_dims, Dim.channel1, Dim.channel2)
                shape = (
                    *shape, *mrq_shapes,
                    mrq_sizes['channel'], mrq_sizes['channel']
                )

            case 'pairwise':

                mrq_dims = [d for d in mrq_sizes if d not in [Dim.k_j, *dims]]
                mrq_shapes = [s for d, s in mrq_sizes.items()
                              if d not in [Dim.k_j, *dims]]

                dims = (*dims, *mrq_dims)
                shape = (*shape, *mrq_shapes)

        # dims q1 q2 j scaling_range channel_left channel_right bootstrap
        self.values = xr.DataArray(np.zeros(shape), dims=dims)
        self.coherence = xr.DataArray(np.zeros(shape[2:]), dims=dims[2:])

        for ind_j, j in enumerate(self.j):

            # 2 - Compute the S_{q1, q2}(j)

            values_mrq1 = mrq1.get_values(j, idx_reject).transpose(
                ..., 'channel')
            dims_mrq1 = values_mrq1.dims
            values_mrq1 = np.abs(values_mrq1.values)

            values_mrq2 = mrq2.get_values(j, idx_reject).transpose(
                ..., 'channel')
            dims_mrq2 = values_mrq2.dims
            values_mrq2 = np.abs(values_mrq2.values)

            pow1 = {
                q1: fast_power(values_mrq1, q1)
                for q1 in self.q1 if q1 != 0
            }

            pow2 = {
                q2: fast_power(values_mrq2, q2)
                for q2 in self.q2 if q2 != 0
            }

            match self.mode:
                case 'all2all':
                    pow1 = {k: v[..., None] for k, v in pow1.items()}
                    pow2 = {k: v[..., None, :] for k, v in pow2.items()}
                    dims_mrq1 = (*dims_mrq1[:-1], Dim.channel1, Dim.channel2)
                    dims_mrq2 = (*dims_mrq2[:-1], Dim.channel1, Dim.channel2)

                case 'pairwise':
                    pass
                    # pow1 = {k: v[..., None] for k, v in pow2.items()}
                    # pow2 = {k: v[..., None] for k, v in pow2.items()}

            for ind_q1, q1 in enumerate(self.q1):
                for ind_q2, q2 in enumerate(self.q2):

                    match (q1, q2):

                        case (0, 0):
                            self.values[ind_q1, ind_q2, ind_j] = 1

                        case (q1, 0):
                            self.values[ind_q1, ind_q2, ind_j] = xr.DataArray(
                                np.log2(np.nanmean(pow1[q1],
                                                   axis=dims_mrq1.index(Dim.k_j))),
                                dims=[d for d in dims_mrq1 if d != Dim.k_j])

                        case (0, q2):
                            self.values[ind_q1, ind_q2, ind_j] = xr.DataArray(
                                np.log2(np.nanmean(pow2[q2],
                                                   axis=dims_mrq2.index(Dim.k_j))),
                                dims=[d for d in dims_mrq2 if d != Dim.k_j])

                        case (q1, q2):
                            self.values[ind_q1, ind_q2, ind_j] = xr.DataArray(
                                np.log2(np.nanmean(pow1[q1] * pow2[q2],
                                                   axis=dims_mrq1.index(Dim.k_j))),
                                dims=[d for d in dims_mrq2 if d != Dim.k_j])

            # 3 - Compute the coherence

            values_mrq1 = mrq1.get_values(j, idx_reject).transpose(
                ..., 'channel')
            dims_mrq1 = values_mrq1.dims
            values_mrq1 = values_mrq1.values

            values_mrq2 = mrq2.get_values(j, idx_reject).transpose(
                ..., 'channel')
            dims_mrq2 = values_mrq2.dims
            values_mrq2 = values_mrq2.values

            match self.mode:
                case 'all2all':
                    values_mrq1 = values_mrq1[..., None]
                    values_mrq2 = values_mrq2[..., None, :]
                    dims_mrq1 = (*dims_mrq1[:-1], Dim.channel1, Dim.channel2)
                    dims_mrq2 = (*dims_mrq2[:-1], Dim.channel1, Dim.channel2)
                case 'pairwise':
                    pass

            self.coherence[ind_j] = xr.DataArray(
                np.nanmean(values_mrq1 * values_mrq2,
                           axis=dims_mrq1.index(Dim.k_j))
                / np.sqrt(
                    np.nanmean(fast_power(values_mrq1, 2) * fast_power(values_mrq2, 2),
                               axis=dims_mrq1.index(Dim.k_j))),
                dims=[d for d in dims_mrq1 if d != Dim.k_j])

        mapping = {
            'j': self.j
        }

        self.coherence.coords.update(mapping)

        mapping |= {
            'q1': self.q1,
            'q2': self.q2,
        }

        self.values.coords.update(mapping)

    def S_qq(self, q1, q2):
        """
        Get bivariate :math:`S_{q_1q_2}` function.
        """
        return self.values[isclose(q1, self.q1), isclose(q2, self.q2)][0]

    def s_qq(self, q1, q2):
        """
        Get bivariate :math:`s_{q_1q_2}` exponents.
        """

        out = self.slope[isclose(q1, self.q1), isclose(q2, self.q2)][0]
        out = out.reshape(len(self.scaling_ranges), *self.values.shape[4:])

        return out

    def plot(self, figsize=None, scaling_range=0, signal_idx1=0,
             signal_idx2=0, plot_CI=True, plot_scales=None, filename=None):
        """
        Plot bivariate structure function.
        """

        if self.mode == 'pairwise':
            signal_idx2 = 0

        if plot_scales is None:
            j1, j2, j_min, j_max = self.get_jrange(None, None, plot_CI)
        else:
            j1, j2, j_min, j_max = self.get_jrange(*plot_scales, plot_CI)

        idx = np.s_[j_min:j_max]

        _, axes = plt.subplots(len(self.q1), len(self.q2), sharex=True,
                               figsize=figsize, squeeze=False)

        x = self.j[idx]

        for ind_q1, q1 in enumerate(self.q1):
            for ind_q2, q2 in enumerate(self.q2):

                y = self.S_qq(q1, q2)[
                    idx, scaling_range, signal_idx1, signal_idx2]

                if self.bootstrapped_obj is not None and plot_CI:

                    __, _, j_min_CI, j_max_CI = \
                        self.bootstrapped_obj.get_jrange(j1, j2)

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
                       ylabel=f'$q_1={q1:.1g}$, $q_2={q2:.1f}$')

                x0, x1 = self.scaling_ranges[scaling_range]

                idx_a = np.s_[ind_q1, ind_q2, scaling_range]
                idx_b = np.s_[signal_idx1, signal_idx2]

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
    """
    Bivariate cumulants.
    """
    n_cumul: int = 2
    m: np.ndarray = field(init=False)
    RHO_MF: np.ndarray = field(init=False)
    rho_mf: float = field(init=False)
    log_cumulants: np.ndarray = field(init=False)

    def __post_init__(self, idx_reject, mrq1, mrq2, min_j):

        if self.n_cumul > 2:
            raise NotImplementedError(
                'Bivariate analysis for cumulant order >= 3 not yet '
                'implemented.')

        super().__post_init__(idx_reject, mrq1, mrq2, min_j)

        if self.bootstrapped_obj is not None:
            self.bootstrapped_obj = self.bootstrapped_obj.structure

        self.margin_m = np.arange(1, self.n_cumul+1)
        self.m = [(m1, m2)
                  for m1 in range(1, self.n_cumul+1)
                  for m2 in range(1, self.n_cumul+1)
                  if m1 + m2 <= self.n_cumul]

        self._compute(mrq1, mrq2, idx_reject)

        self.slope = np.zeros(
            (self.n_cumul+1, self.n_cumul+1, len(self.scaling_ranges),
             *self.values.shape[4:]))
        self.intercept = np.ones_like(self.slope)
        self.weights = np.ones_like(self.slope)

        idx_margin1 = np.s_[1:, 0]
        idx_margin2 = np.s_[0, 1:]

        slope1, intercept1, _ = self._compute_fit(
            mrq1, mrq2, margin=0, value_name='margin1_values')
        self.slope[idx_margin1] = slope1[:, 0, :, :, None]
        self.intercept[idx_margin1] = intercept1[:, 0, :, :, None]

        slope2, intercept2, _ = self._compute_fit(
            mrq1, mrq2, margin=1, value_name='margin2_values')
        self.slope[idx_margin2] = slope2[:, 0, :, None]
        self.intercept[idx_margin2] = intercept2[:, 0, :, None]

        idx = np.s_[1:, 1:]
        self.slope[idx], self.intercept[idx], _ = self._compute_fit(mrq1, mrq2)

        # self.margin1_log_cumulants = self.margin1_slope * np.log2(np.e)
        # self.margin2_log_cumulants = self.margin2_slope * np.log2(np.e)

        self.log_cumulants = self.slope * np.log2(np.e)

        self._compute_rho()

    def _compute(self, mrq1, mrq2, idx_reject):

        n_rep = self._prepare_nrep(mrq1, mrq2)

        self.margin1_values = np.zeros(
            (self.n_cumul, 1, len(self.j), len(self.scaling_ranges),
             n_rep[0], *n_rep[2:])
        )
        self.margin2_values = np.zeros(
            (self.n_cumul, 1, len(self.j), len(self.scaling_ranges),
             n_rep[1], *n_rep[2:])
        )
        self.values = np.zeros((
            len(self.m), 1, len(self.j),
            len(self.scaling_ranges), *n_rep))

        moments = np.zeros_like(self.values)
        margin1_moments = np.zeros_like(self.margin1_values)
        margin2_moments = np.zeros_like(self.margin2_values)

        for ind_j, j in enumerate(self.j):

            pow1 = {
                m: fast_power(np.log(np.abs(
                    mrq1.get_values(j, idx_reject))), m)
                for m in self.margin_m
            }
            pow2 = {
                m: fast_power(np.log(np.abs(
                    mrq2.get_values(j, idx_reject))), m)
                for m in self.margin_m
            }

            # Compute margins
            for ind_m, m in enumerate(self.margin_m):

                margin1_moments[ind_m, 0, ind_j] = np.nanmean(pow1[m], axis=0)
                margin2_moments[ind_m, 0, ind_j] = np.nanmean(pow2[m], axis=0)

                # Margin of mrq1
                aux = 0

                for ind_n, n in enumerate(np.arange(1, m)):

                    aux += (special.binom(m-1, n-1)
                            * self.margin1_values[ind_n, 0, ind_j]
                            * margin1_moments[ind_m-ind_n-1, 0, ind_j]
                            )

                self.margin1_values[ind_m, 0, ind_j] = \
                    margin1_moments[ind_m, 0, ind_j] - aux

                # Margin of mrq2
                aux = 0

                for ind_n, n in enumerate(np.arange(1, m)):

                    aux += (special.binom(m-1, n-1)
                            * self.margin2_values[ind_n, 0, ind_j]
                            * margin2_moments[ind_m-ind_n-1, 0, ind_j]
                            )

                self.margin2_values[ind_m, 0, ind_j] = \
                    margin2_moments[ind_m, 0, ind_j] - aux

            # Compute bivariate cumulants
            pow1 = {k: v[..., None] for k, v in pow1.items()}

            match self.mode:
                case 'all2all':
                    pow2 = {k: v[..., None, :] for k, v in pow2.items()}
                case 'pairwise':
                    pow2 = {k: v[..., None] for k, v in pow2.items()}

            for ind_m, (m1, m2) in enumerate(self.m):

                moments[ind_m, 0, ind_j] = np.nanmean(
                    (pow1[m1] * pow2[m2]), axis=0
                )

                ind_m1 = list(self.margin_m).index(m1)
                ind_m2 = list(self.margin_m).index(m2)

                if m1 == m2 == 1:

                    self.values[ind_m, 0, ind_j] = (
                        moments[ind_m, 0, ind_j]
                        - self.margin1_values[ind_m1, 0, ind_j][:, :, None]
                        * self.margin2_values[ind_m2, 0, ind_j][:, None]
                    )

    def _compute_rho(self):

        if self.formalism == 'wavelet coef':
            self.RHO_MF = None
            self.rho_mf = None
            return

        self.RHO_MF = (self.C11 / np.abs(np.sqrt(self.C02 * self.C20)))
        self.rho_mf = -self.c11 / np.abs(np.sqrt(self.c02 * self.c20))

    def __getattr__(self, name):

        match tuple(name):

            case ['c', m1, m2] if m1.isdigit() and m2.isdigit():

                m1, m2 = int(m1), int(m2)

                return self.log_cumulants[m1, m2]

            case ['C', '0', '0']:
                return np.ones(self.values.shape[2:])

            case ['C', m1, '0'] if m1.isdigit():
                return self.margin1_values[int(m1)-1, 0]

            case ['C', '0', m2] if m2.isdigit():
                return self.margin2_values[int(m2)-1, 0]

            case ['C', m1, m2] if m1.isdigit() and m2.isdigit():

                m1, m2 = int(m1), int(m2)
                if (m1, m2) not in self.m:
                    raise ValueError(
                        f'Cumulant of order {m1}, {m2} has not been computed')

                return self.values[self.m.index((m1, m2)), 0]

        # if name[0] == 'c' and len(name) == 3 and name[1:].isdigit():

        #     out = self.log_cumulants[self.m == int(name[1]),
        #                              self.m == int(name[2])][0]

        #     return out.reshape(
        #         len(self.scaling_ranges), *self.values.shape[4:])

        # if name[0] == 'C' and len(name) == 3 and name[1:].isdigit():
        #     return self.values[self.m == int(name[1]),
        #                        self.m == int(name[2])][0]

        if (super_attr := super().__getattr__(name)) is not None:
            return super_attr

        return super().__getattribute__(name)

    def plot(self, j1=None, j2=None, figsize=None, scaling_range=0,
             filename=None, signal_idx1=1, signal_idx2=0, plot_CI=False,
             **kwargs):
        """
        Plot bivariate cumulants.
        """

        # if self.j.min() > j1:
        #     raise ValueError(f"Expected mrq to have minium scale {j1=}, got "
        #                      f"{self.j.min()} instead")

        ncol = self.n_cumul + 1

        if figsize is None:
            figsize = (3.3 * ncol, 3.3 * ncol)

        fig, axes = plt.subplots(ncol,
                                 ncol,
                                 squeeze=False,
                                 figsize=figsize,
                                 sharex=True)

        fig.suptitle(self.formalism + r" - bivariate cumulants $C_{m, m'}(j)$")

        for m1 in range(self.n_cumul + 1):
            for m2 in range(self.n_cumul + 1):

                if m1 != 0 and m2 != 0 and (m1, m2) not in self.m:
                    fig.delaxes(axes[m1][m2])
                    continue

                plot_bicm(
                    self, m1, m2, j1, j2, scaling_range=scaling_range,
                    ax=axes[m1][m2], plot_legend=True,
                    signal_idx1=signal_idx1, signal_idx2=signal_idx2,
                    **kwargs)

        if filename is not None:
            plt.savefig(filename)

    def _compute_legendre(self, h_support=(0, 1.5), resolution=100,
                          signal_idx1=0, signal_idx2=1, idx_range=0):
        """
        Compute the bivariate Legendre spectrum.
        """

        if signal_idx1 == signal_idx2:
            raise ValueError('signal_idx1 should be different from signal_idx2')

        h_support = np.linspace(*h_support, resolution)

        sl_ = np.s_[idx_range, signal_idx1, signal_idx2]

        c11 = self.c11[sl_]
        c10 = self.c10[sl_]
        c01 = self.c01[sl_]
        c20 = self.c20[sl_]
        c02 = self.c02[sl_]

        b = (c20 * c02) - (c11 ** 2)

        L = np.ones((resolution, resolution))

        for i, h in enumerate(h_support):
            L[i, :] += c02 * b / 2 * (((h - c10) / b) ** 2)
            L[:, i] += c20 * b / 2 * (((h - c01) / b) ** 2)

        for i, h1 in enumerate(h_support):
            for j, h2 in enumerate(h_support):
                L[i, j] -= (c11 * b
                            * ((h1 - c10) / b)
                            * ((h2 - c01) / b))

        return h_support, L

    def plot_legendre(self, h_support=(0, 1.5), resolution=100,
                      figsize=(10, 10), cmap=None, signal_idx1=0,
                      signal_idx2=1, idx_range=0):
        """
        Plot the bivariate Legendre spectrum
        """

        h, L = self._compute_legendre(
            h_support=h_support, resolution=200, signal_idx1=signal_idx1,
            signal_idx2=signal_idx2, idx_range=idx_range)

        h_x = h[L.max(axis=0) >= 0]
        h_y = h[L.max(axis=1) >= 0]

        hmin = min([*h_x, *h_y])
        hmax = max([*h_x, *h_y])

        h, L = self._compute_legendre((hmin, hmax), resolution)

        cmap = cmap or plt.cm.viridis  # pylint: disable=no-member

        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw={'projection': '3d'})
        # ax = fig.add_subplot(1, 1, 1, projection='3d')

        ax.set_proj_type('persp', focal_length=.15)

        h_x = h[L.max(axis=0) >= 0]
        h_y = h[L.max(axis=1) >= 0]

        L = L[:, L.max(axis=0) >= 0]
        L = L[L.max(axis=1) >= 0, :]

        L[L < 0] = None

        colors = cmap(L)
        colors[L < 0] = 0

        X, Y = np.meshgrid(h_x, h_y)

        light = mpl.colors.LightSource(azdeg=60, altdeg=60)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, L, alpha=.95, cmap=cmap,
                            #   facecolors=colors,
                            # lightsource=light,
                            #    linewidth=1, vmin=0, vmax=1,
                            #    rstride=1, cstride=1,
                            # linestyle='-',
                            #    zorder=1)
        )

        # argmax = np.argmax(L, axis=-1)
        # ax.contour(X[..., argmax], Y[..., argmax], L[..., argmax], zdir='x')

        # ax.contour(X, Y, L > 1, zdir='x', offset=h_x.min(), levels=0)

        ax.set(xlabel='$h_1$', ylabel='$h_2$',
               zlabel=r'$\mathcal{L}(h_1, h_2)$')

        # ax.contour(X, Y, L, zdir='y', offset=h_y.max(), levels=[0])

        # ax.contour(X, Y, L, zdir='x', levels=10)
        # ax.contour(X, Y, L, zdir='y', levels=10)
        # ax.contour(X, Y, L, zdir='z', levels=10)

        # surf.set_edgecolors((0.1, 0.2, 0.5, 1))
        ax.set_zlim(0, 1)
        ax.view_init(elev=45)

        # TODO manage to plot the contours or switch to 3D plotting libs
        # fig.colorbar(surf, shrink=0.6, aspect=10)

    def plot_legendre_pv(self, resolution=30, figsize=(10, 10), cmap=None,
                         use_ipyvtk=False):
        """
        Plot the bivariate Legendre spectrum using pyvista.
        """

        import pyvista as pv

        if use_ipyvtk:
            from ..viz import start_xvfb  # pylint: disable=C0415
            start_xvfb()

        h, L = self._compute_legendre(resolution=200)

        h_x = h[L.max(axis=0) >= 0]
        h_y = h[L.max(axis=1) >= 0]

        hmin = min([*h_x, *h_y])
        hmax = max([*h_x, *h_y])

        h, L = self._compute_legendre((hmin, hmax), resolution)

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
