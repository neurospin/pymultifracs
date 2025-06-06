"""
Authors: Merlin Dumeur <merlin@dumeur.net>
         Omar D. Domingues <omar.darwiche-domingues@inria.fr>
"""

# pylint: disable=W0221

from dataclasses import dataclass, field, InitVar
import inspect

import numpy as np
import xarray as xr
from scipy import special

import matplotlib.pyplot as plt

# from .multiresquantity import WaveletDec
from .regression import prepare_weights, prepare_regression, \
    linear_regression, compute_R2, compute_RMSE
from .autorange import compute_Lambda, compute_R, find_max_lambda
from .utils import fast_power, mask_reject, isclose, fixednansum, \
    AbstractDataclass, Formalism, Dim, _expand_align
from . import multiresquantity, viz


@dataclass(kw_only=True)
class AbstractScalingFunction(AbstractDataclass):
    """
    Abstract class for general scaling functions
    """
    scaling_ranges: list[tuple[int]]
    idx_reject: InitVar[dict[int, np.ndarray] | None] = None
    weighted: str | None = None
    n_channel: int = field(init=False)
    j: np.ndarray = field(init=False)
    formalism: Formalism = field(init=False)
    nj: dict[int, np.ndarray] = field(init=False, repr=False)
    values: np.ndarray = field(init=False, repr=False)
    slope: np.ndarray = field(init=False, repr=False)
    intercept: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False)

    @classmethod
    def _from_dict(cls, d):
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

    def get_nj_interv(self, j_min, j_max):
        """
        Returns the number of coefficients on an interval of temporal scales.
        """
        return self.nj[j_min-min(self.j):j_max-min(self.j)+1]

    def _get_j_min_max(self):

        j_min = min(sr[0] for sr in self.scaling_ranges)
        j_max = max(sr[1] for sr in self.scaling_ranges)

        return j_min, j_max

    def __getattr__(self, name):

        # if name == 'n_rep':
        #     return self.intercept.shape[-1]

        if (super_attr := super().__getattr__(name)) is not None:
            return super_attr

        return self.__getattribute__(name)


@dataclass(kw_only=True)
class ScalingFunction(AbstractScalingFunction):
    """"
    General DWT-based scaling function.
    """
    mrq: InitVar[multiresquantity.WaveletDec]
    min_j: InitVar[int] = 1
    variable_suffix: str = field(init=False)
    regularity_suffix: str = field(init=False)
    gamint: float = field(init=False)

    def __post_init__(self, idx_reject, mrq, min_j):  # pylint: disable=W0613

        self.gamint = mrq.gamint
        self.n_channel = mrq.n_channel
        self.formalism = mrq.get_formalism()
        self.variable_suffix, self.regularity_suffix = mrq._get_suffix()
        self.j = np.array(list(mrq.values))
        self.j = self.j[self.j >= min_j]

        self.nj = mrq.get_nj_interv(min_j, idx_reject=idx_reject)

    def compute_R2(self):
        """
        Computes :math:`R^2` for the estimated linear regressions.
        """
        return compute_R2(
            self.values, self.slope, self.intercept, self.weights,
            [self._get_j_min_max()], self.j)

    def compute_RMSE(self):
        """
        Computes root mean square error for the estimated linear regressions.
        """
        return compute_RMSE(
            self.values, self.slope, self.intercept, self.weights,
            [self._get_j_min_max()], self.j)

    def compute_R(self):
        """
        Computes :math:`R` for bootstrap-based automated range selection.
        """

        # values = self.values.reshape(
        #     *self.values.shape[:3], self.n_channel, -1)
        # slope = self.slope.reshape(*self.slope.shape[:2], self.n_channel, -1)
        # intercept = self.intercept.reshape(
        #     *self.intercept.shape[:2], self.n_channel, -1)

        # if self.weights.shape[-1] > 1:
        #     weights = self.weights.reshape(
        #         *self.weights.shape[:3], self.n_channel, -1)
        # else:
        #     weights = self.weights[..., None]

        return compute_R(self.values, self.slope, self.intercept, self.weights,
                         [self._get_j_min_max()], self.j)

    def compute_Lambda(self):
        """
        Computes :math:`\\Lambda` for bootstrap-based automated range
        selection. [1]_

        References
        ----------

        .. [1] https://doi.org/10.1016/j.sigpro.2014.06.002

        """

        R = self.compute_R()
        R_b = self.bootstrapped_obj.compute_R()

        return compute_Lambda(R, R_b)

    def find_best_range(self, per_moment=False):
        """
        Find the best range among those computed, given bootstrap was already
        performed

        Parameters
        ----------
        per_moment : bool
            If True, returns the best range for each moment. Otherwise, by
            default returns the overall best range for all moments.

        References
        ----------

        .. [1] https://doi.org/10.1016/j.sigpro.2014.06.002
        """
        return find_max_lambda(self.compute_Lambda(), per_moment)

    def get_jrange(self, j1=None, j2=None, bootstrap=False):
        """
        Sanitize the scaling range :math:`[j_1, j_2]` and find the associated
        indices in the ``mrq.j`` array.
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

    def _compute_fit(self, value_name=None, out_name=None):

        if value_name is not None:
            values = getattr(self, value_name)
        else:
            values = self.values

        # slope = np.zeros(
        #     (values.shape[0], len(self.scaling_ranges),
        #      values.shape[-1]))

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j, values.dims)

        # self.intercept = np.zeros_like(slope)

        y = values.sel(j=slice(j_min, j_max))

        if self.weighted == 'bootstrap':

            if self.bootstrapped_obj is None:
                std = self.std_values().sel(j=slice(j_min, j_max))

            else:

                if j_min < self.bootstrapped_obj.j.min():
                    raise ValueError(
                        f"Bootstrap minimum scale "
                        f"{self.bootstrapped_obj.j.min()} inferior to minimum "
                        f"scale {j_min} used in estimation")

                # std_slice = np.s_[
                #     int(j_min - self.bootstrapped_obj.j.min()):
                #     int(j_max - self.bootstrapped_obj.j.min() + 1)]

                std = self.bootstrapped_obj.std_values().sel(j=slice(j_min, j_max))

        else:
            std = None

        self.weights = prepare_weights(
            self.get_nj_interv, self.weighted, n_ranges, j_min, j_max,
            self.scaling_ranges, y, std)

        # nan_weighting = np.ones_like(y)
        # nan_weighting[np.isnan(y)] = np.nan

        # self.weights *= nan_weighting

        slope, self.intercept = linear_regression(x, y, self.weights)

        if out_name is not None:
            slope = setattr(self, out_name, slope)
        else:
            self.slope = slope


@dataclass(kw_only=True)
class StructureFunction(ScalingFunction):
    """
    Contains the structre functions and their linear fit.

    .. note:: Should not be instanciated but instead obtained from calling
        :func:`pymultifracs.mfa`

    Attributes
    ----------
    j : ndarray, shape (n_j)
        Values of j covered by the analysis.
    nj : dict[int, ndarray]
        Dictionnary giving the number of non-NaN values at every scale. Array
        are of shape (n_rep)
    gamint : float
        Value of gamint used in the computation of the underlying MRQ.
    formalism : str
        Formalism used. Can be any of 'wavelet coefs', 'wavelet leaders',
        or 'wavelet p-leaders'.
    n_channel : int
        Number of underlying signals in the wavelet decomposition. May not
        match the dimensionality of the values arrays in case there are
        multiple repetitions associated to a single signal.
    q : ndarray, shape (n_moments)
        :math:`q` values for which the structure functions are computed.
    values : ndarray, shape (n_moments, n_j, n_scaling_ranges, n_rep)
        :math:`S_q(j, k)`.
    scaling_ranges : list[tuple[int, int]]
        List of pairs of scales :math:`(j_1, j_2)` delimiting the temporal
        scale support over which the estimates are regressed.
    slope : ndarray, shape (n_moments, n_scaling_ranges, n_rep)
        :math:`s_q`.
    H : ndarray
        :math:`H = s_2 / 2`.
    intercept : ndarray, shape (n_moments, n_scaling_ranges, n_rep)
        Intercept of the linear regression.
    weighted : str | None
        Weighting mode for the linear regressions. Defaults to None, which is
        no weighting. Possible values are 'Nj' which weighs by number of
        coefficients, and 'bootstrap' which weights by bootstrap-derived
        estimates of variance.
    weights : ndarray
        Weights of the linear regression.
    bootstrapped_obj : StructureFunction | None
        Storing the bootstrapped version of the structure function if
        bootstraping has been used.
    """
    q: np.ndarray
    H: np.ndarray = field(init=False)

    def __post_init__(self, idx_reject, mrq, min_j):

        super().__post_init__(idx_reject, mrq, min_j)

        if self.bootstrapped_obj is not None:
            self.bootstrapped_obj = self.bootstrapped_obj.structure

        self.dims = (
            Dim.q, Dim.j, Dim.scaling_range, *mrq.dims[1:]
        )

        self._compute(mrq, idx_reject)
        self._compute_fit()

    def _compute(self, mrq, idx_reject):

        shape = (
            len(self.q), len(self.j), len(self.scaling_ranges),
        )
        dims = (Dim.q, Dim.j, Dim.scaling_range)

        mrq_sizes = mrq.get_values(self.j.max()).sizes

        mrq_dims = [d for d in mrq_sizes if d not in [Dim.k_j, *dims]]
        mrq_shapes = [s for d, s in mrq_sizes.items()
                      if d not in [Dim.k_j, *dims]]

        dims = (*dims, *mrq_dims)
        shape = (*shape, *mrq_shapes)

        coords = {
            'q': self.q,
            'j': self.j
        }

        # dims q1 q2 j scaling_range channel_left channel_right bootstrap
        self.values = xr.DataArray(np.zeros(shape), dims=dims, coords=coords)

        for ind_j, j in enumerate(self.j):

            c_j = mrq.get_values(j, idx_reject)

            mask_nan = np.isnan(c_j) | np.isinf(c_j)
            N_useful = (~mask_nan.values).sum(axis=c_j.dims.index(Dim.k_j))
            idx_unreliable = N_useful < 3

            # c_j = _expand_align(c_j, reference_order=self.dims[1:])

            for ind_q, q in enumerate(self.q):

                self.values[ind_q, ind_j] = xr.DataArray(
                    np.log2(np.nanmean(fast_power(np.abs(c_j.values), q),
                                       axis=c_j.dims.index(Dim.k_j))),
                    dims=[d for d in c_j.dims if d != Dim.k_j])

                # if idx_unreliable.any():

                    # self.values[ind_q, ind_j] = np.nan
                    # for i in range(idx_unreliable.sizes[Dim.channel]):

                    # np.nan

            if idx_unreliable.any():
                # idx_unreliable = _expand_align(idx_unreliable, reference_order=self.dims[1:])
                self.values.values[:, ind_j, ..., idx_unreliable] = np.nan

        # self.values.where(np.isinf(self.values), np.nan)
        self.values.values[np.isinf(self.values)] = np.nan

        # print(self.values)

    def _get_H(self):
        return self.slope.sel(q=2) / 2

    def S_q(self, q):
        """
        Returns :math:`S_q(j)` for given ``q``.
        """
        return self.values.sel(q=q, method='nearest', tolerance=.1)

    def s_q(self, q):
        """
        Returns :math:`s_q` for given ``q``.
        """
        return self.slope.sel(q=q, method='nearest', tolerance=.1)

    def __getattr__(self, name):

        if name == 'H':
            return self._get_H()

        # if name == 'S2':
        #     out = self.values[self.q == 2]
        #     return out.reshape(out.shape[0], self.n_channel, -1)

        if name == 'zeta':
            return self.slope

        if (super_attr := super().__getattr__(name)) is not None:
            return super_attr

        return self.__getattribute__(name)

    def plot(self, nrow=4, filename=None, ignore_q0=True, figsize=None,
             scaling_range=0, plot_scales=None, plot_CI=True, signal_idx=0):
        """
        Plots the structure functions.

        Parameters
        ----------

        nrow : int
            Number of rows in the plot.
        filename : str | None
            If not None, the file is saved to `filename`
        ignore_q0 : bool
            Whether to include the structure function for :math:`q=0`, which
            is always going to be a constant function valued 1. Defaults to
            True.
        figsize : tuple[int, int] | None
            Size of the figure, in inches.
        scaling_range : int
            If multiple scaling ranges were used in fitting, indicates the
            index to use.
        plot_scales : tuple[int, int] | None
            Takes a tuple of the form :math:`(j_1, j_2)`: Constrains the
            x-axis to the interval :math:`[j_1, j_2]`.
        plot_CI : bool
            If using bootstrap, plot bootstrap-derived confidence interval
            on the structure function.
        signal_idx : int
            If using a multivariate signal, index of the signal to plot.
        """

        if plot_scales is None:
            j1, j2, j_min, j_max = self.get_jrange(None, None, plot_CI)
        else:
            j1, j2, j_min, j_max = self.get_jrange(*plot_scales, plot_CI)

        idx = np.s_[j_min:j_max]

        # if self.n_rep > 1:
        #     raise ValueError('Cannot plot structure functions for more than '
        #                      '1 repetition at a time')

        nrow = min(nrow, len(self.q))
        nq = len(self.q) + (-1 if 0.0 in self.q and ignore_q0 else 0)

        if nq > 1:
            plot_dim_1 = nrow
            plot_dim_2 = int(np.ceil(nq / nrow))

        else:
            plot_dim_1 = 1
            plot_dim_2 = 1

        fig, axes = plt.subplots(plot_dim_1,
                                 plot_dim_2,
                                 squeeze=False,
                                 figsize=figsize,
                                 sharex=True,
                                 layout='tight')

        # fig.suptitle(self.formalism +
        #              r' - structure functions $\log_2(S(j,q))$')

        x = self.j[idx]

        counter = 0

        for ind_q, q in enumerate(self.q):

            if q == 0.0 and ignore_q0:
                continue

            y = self.S_q(q).sel(j=slice(j1, j2)).isel(
                scaling_range=scaling_range, channel=signal_idx)

            if self.bootstrapped_obj is not None and plot_CI:

                j_min_CI, j_max_CI, *_ = self.bootstrapped_obj.get_jrange(
                    j1, j2)

                CI = self.CIE_S_q(q).sel(j=slice(j_min_CI, j_max_CI)).isel(
                    scaling_range=scaling_range, channel=signal_idx)

                CI -= y#[:, None]
                CI[:, 1] *= -1
                assert (CI < 0).sum() == 0
                CI = CI.transpose()

            else:
                CI = None

            ax = axes[counter % nrow][counter // nrow]
            ax.errorbar(x, y, CI, fmt='r--.', zorder=4)
            ax.set_xlabel('Temporal scale $j$')
            ax.set_ylabel(f'$S_{{{q:.1g}}}{self.variable_suffix}(j)$')
            ax.tick_params(bottom=False, top=False, which='minor')

            counter += 1

            x0, x1 = self.scaling_ranges[scaling_range]
            slope = self.slope[ind_q, scaling_range, 0]
            intercept = self.intercept[ind_q, scaling_range, 0]

            assert x0 in x, "Scaling range not included in plotting range"
            assert x1 in x, "Scaling range not included in plotting range"

            y0 = slope*x0 + intercept
            y1 = slope*x1 + intercept

            if self.bootstrapped_obj is not None and plot_CI:
                CI = self.CIE_s_q(q)[scaling_range, signal_idx]
                CI_legend = f"; [{CI[0]:.1f}, {CI[1]:.1f}]"
            else:
                CI_legend = ""

            legend = (rf'$s_{{{q:.1g}}}{self.variable_suffix}$ = {slope:.2f}'
                      + CI_legend)

            ax.plot([x0, x1], [y0, y1], color='k',
                    linestyle='-', linewidth=2, label=legend, zorder=5)
            ax.legend()

        for j in range(counter, len(axes.flat)):
            fig.delaxes(axes[j % nrow][j // nrow])

        if filename is not None:
            plt.savefig(filename)

    def plot_scaling(self, filename=None, ax=None, signal_idx=0, range_idx=0,
                     **plot_kw):
        """
        Plots the scaling function :math:`\\zeta(q)`.

        Parameters
        ----------

        filename : str | None
            If not None, saves the figure to `filename`.
        ax : Axes | None
            Provides the axes on which to draw the function.
            Defaults to None, which creates a new figure.
        signal_idx : int
            If using a multivariate signal, index of the signal to plot.
        range_idx : int
            If multiple scaling ranges were used in fitting, indicates the
            index to use.
        **plot_kw : dict
            Extra arguments forwarded to the plot function call.
        """

        assert len(self.q) > 1, ("This plot is only possible if more than 1 q",
                                 " value is used")

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 2.5), layout='tight')

        ax.plot(self.q, self.slope[:, range_idx, signal_idx], **plot_kw)

        ax.set(
            xlabel='Moment $q$',
            ylabel=rf'Scaling function $\zeta{self.variable_suffix}(q)$',
            # title=self.formalism + ' - scaling function'
            )

        # plt.draw()

        if filename is not None:
            plt.savefig(filename)


@dataclass(kw_only=True)
class Cumulants(ScalingFunction):
    r"""
    Computes and analyzes cumulant.

    .. note:: Should not be instanciated but instead obtained from calling
        :func:`pymultifracs.mfa`

    Attributes
    ----------
    j : ndarray of int, shape (n_j,)
        List of the j values (scales), in order presented in the value arrays.
    nj : ndarray of int, shape (n_j,)
        Dictionnary giving the number of non-NaN values at every scale. Arrays
        are of the shape (n_rep,).
    gamint : float
        Value of gamint used in the computation of the underlying MRQ.
    formalism : str
        Formalism used. Can be any of: 'wavelet coefs', 'wavelet leaders',
        'wavelet p-leaders', or 'weak scaling exponent'.
    n_channel : int
        Number of underlying signals in the wavelet decomposition. May not
        match the dimensionality of the values arrays (n_rep) in case there are
        multiple repetitions associated to a single signal, for instance in
        bootstrapping.
    n_cumul : int
        Maximum order of the computed cumulants.
    m : ndarray of int, shape (n_cumul,)
        Cumulant order values :math:`m`, in the order used internally.
    values : ndarray of float, shape (n_cumulants, n_scales, n_rep)
        :math:`C_m(j)`.
    scaling_ranges : List[(int, int)]
        List of pairs of scales delimiting the temporal scale support over
        which the estimates are regressed.
    log_cumulants : ndarray, shape (n_cumulants, n_rep)
        :math:`(c_m)_m`, slopes of the curves :math:`j \times C_m(j)`.
    var_log_cumulants : ndarray, shape (n_cumulants, n_rep)
        Estimates of the variance of log-cumulants.
    weighted : str | None
        Weighting mode for the linear regressions. Defaults to None, which
        means no weighting. Possible values are ``'Nj'`` which weighs by number
        of coefficients, and 'bootstrap' which weights by bootstrap-derived
        estimates of variance.
    weights : ndarray of float, shape () #TODO: plot shape of weights here
        Weights of the linear regression.
    bootstrapped_obj : Cumulants | None
        Storing the bootstrapped version of the structure function if
        bootstraping has been used.

    See Also
    --------

    Examples
    --------
    """
    n_cumul: int
    robust: InitVar[bool] = False
    robust_kwargs: InitVar[dict[str, object]] = dict()
    m: np.ndarray = field(init=False)
    log_cumulants: np.ndarray = field(init=False)

    def __post_init__(self, idx_reject, mrq, min_j, robust, robust_kwargs):

        super().__post_init__(idx_reject, mrq, min_j)

        if self.bootstrapped_obj is not None:
            self.bootstrapped_obj = self.bootstrapped_obj.cumulants

        self.m = np.arange(1, self.n_cumul+1)

        dims = (Dim.m, Dim.j, Dim.scaling_range)
        shape = (len(self.m), len(self.j), len(self.scaling_ranges))

        mrq_dims, mrq_shapes = [], []

        for d, s, in mrq.get_values(self.j.max()).sizes.items():

            if d in [Dim.k_j, *dims]:
                continue

            mrq_dims.append(d)
            mrq_shapes.append(s)

        self.values = xr.DataArray(
            np.zeros((*shape, *mrq_shapes)),
            dims=(*dims, *mrq_dims),
            coords={Dim.j: self.j, Dim.m: self.m})

        if robust:
            self._compute_robust(mrq, idx_reject, **robust_kwargs)
        else:
            self._compute(mrq, idx_reject)

        self._compute_fit()
        self.log_cumulants = self.slope * np.log2(np.e)

    def __repr__(self):

        out = "Cumulants"
        display_params = (
            'scaling_ranges weighted n_cumul').split(' ')

        for param in display_params:
            out += f" {param} = {getattr(self, param)}"

        return out

    def _compute_robust(self, mrq, idx_reject):

        # moments = np.zeros_like(self.values)
        # aux = np.zeros_like(moments)

        from . import robust  # pylint: disable=C0415

        for ind_j, j in enumerate(self.j):

            T_X_j = np.abs(mrq.get_values(j))
            dims = T_X_j.dims
            T_X_j = T_X_j.values
            # T_X_j = T_X_j[:, None, :]

            # if self.formalism == 'wavelet p-leader':
            #     T_X_j = T_X_j * mrq.ZPJCorr[None, :, :, ind_j]

            np.log(T_X_j, out=T_X_j)

            # dropping infinite coefsx
            # T_X_j[np.isinf(T_X_j)] = np.nan

            mask_nan = np.isnan(T_X_j)
            mask_nan |= np.isinf(T_X_j)

            if idx_reject is not None and j in idx_reject:
                mask_nan |= idx_reject[j]

            T_X_j[mask_nan] = np.nan

            # log_T_X_j = mask_reject(
            #     log_T_X_j, idx_reject, j, mrq.interval_size)

            values = robust.compute_robust_cumulants(
                T_X_j, dims, self.m, **self.robust_kwargs)

            self.values.loc[{Dim.j: j}] = values

    def _compute(self, mrq, idx_reject):

        moments = xr.zeros_like(self.values)

        for ind_j, j in enumerate(self.j):

            T_X_j = np.abs(mrq.get_values(j, None))
            dims = T_X_j.dims
            T_X_j = T_X_j.values

            np.log(T_X_j, out=T_X_j)

            mask_nan = np.isnan(T_X_j)
            mask_nan |= np.isinf(T_X_j)

            if idx_reject is not None and j in idx_reject:
                # delta = (mrq.interval_size - 1) // 2
                mask_nan |= idx_reject[j]

            T_X_j[mask_nan] = 0

            N_useful = (~mask_nan).sum(axis=dims.index(Dim.k_j))
            idx_unreliable = N_useful < 3

            for m in self.m:

                loc_dict = {Dim.m: m, Dim.j: j}

                moments.loc[loc_dict] = xr.DataArray(
                    np.sum(fast_power(T_X_j, m),
                           axis=dims.index(Dim.k_j)) / N_useful,
                    dims=[d for d in dims if d != Dim.k_j]
                )
                # np.divide(
                #     moments[ind_m, ind_j], N_useful, out=moments[ind_m, ind_j])

                # idx_unreliable = N_useful < 3

                # for i in range(idx_unreliable.shape[0]):
                #     moments[ind_m, ind_j, i, idx_unreliable[i]] = np.nan

                if m == 1:
                    self.values.loc[loc_dict] = moments.sel(m=m, j=j)
                else:
                    aux = 0

                    for n in np.arange(1, m):
                        aux += (special.binom(m-1, n-1)
                                * self.values.sel(m=n, j=j)
                                * moments.sel(m=m-n, j=j))

                    self.values.loc[loc_dict] = moments.sel(m=m, j=j) - aux

            if idx_unreliable.any():
                self.values.values[:, ind_j, ..., idx_unreliable] = np.nan
                    # for i in range(idx_unreliable.shape[0]):
                    #     self.values[ind_m, ind_j, :, idx_unreliable[i]] = \
                    #         np.nan

        # self.values.where(np.isinf(self.values), np.nan)
        self.values.values[np.isinf(self.values)] = np.nan

    def __getattr__(self, name):

        if name[0] == 'c' and len(name) == 2 and name[1:].isdigit():
            return self.log_cumulants.sel(m=int(name[1]))

        if name[0] == 'C' and len(name) == 2 and name[1:].isdigit():
            return self.values.sel(m=int(name[1]))

        if name == 'M':
            return -self.c2

        if (super_attr := super().__getattr__(name)) is not None:
            return super_attr

        return self.__getattribute__(name)

    def plot(self, figsize=None, nrow=3, j1=None, filename=None,
             range_idx=0, n_cumul=None, signal_idx=0, **kwargs):
        """
        Plots the :math:`C_m(j)` and their associated :math:`c_m` fits.

        Parameters
        ----------

        figsize: (int, int) | None
            If not None, indicates the size of the figure.
        nrow : int
            Number of rows of the figure.
        j1 : int
            Constrains the plot to scales :math:`j \\geq j_1`.
        filename : str | None
            If not None, saves the figure to ``filename``.
        signal_idx : int
            If using a multivariate signal, index of the signal to plot.
        range_idx : int
            If multiple scaling ranges were used in fitting, indicates the
            index to use.
        **kwargs : dict
            Optional arguments sent to :func:`pymultifracs.viz.plot_cumulants`.
        """

        if n_cumul is None:
            n_cumul = self.n_cumul

        if n_cumul > self.n_cumul:
            raise ValueError(
                'Cannot plot more cumulants than were computed '
                f'({self.n_cumul})'
            )

        return viz.plot_cumulants(
            self, figsize=figsize, nrow=nrow, j1=j1, filename=filename,
            range_idx=range_idx, n_cumul=n_cumul, signal_idx=signal_idx,
            **kwargs)


@dataclass(kw_only=True)
class MFSpectrum(ScalingFunction):
    """
    Estimates the Multifractal Spectrum

    Based on equations 2.74 - 2.78 of Herwig Wendt's thesis [1]_

    .. note:: Should not be instanciated but instead obtained from calling
        :func:`pymultifracs.mfa`

    Attributes
    ----------
    formalism : str
        Formalism used. Can be any of 'wavelet coefs', 'wavelet leaders',
        or 'wavelet p-leaders'.
    j : ndarray, shape (n_scales,)
        List of the j values (scales), in order presented in the value arrays.
    nj : dict[int, ndarray]
        Dictionnary giving the number of non-NaN values at every scale. Array
        are of shape (n_rep,)
    gamint : float
        Value of gamint used in the computation of the underlying MRQ.
    formalism : str
        Formalism used. Can be any of 'wavelet coefs', 'wavelet leaders',
        or 'wavelet p-leadParameters
        ----------

        figsize: (int, int) | None
            If not None, indicates the size of the figure.
        nrow : int
            Number of rows of the figure.
        j1 : int
            Constrains the plot to scales :math:`j \\geq j_1`.
        filename : str | None
            If not None, saves the figure to ``filename``.
        signal_idx : int
            If using a multivariate signal, index of the signal to plot.
        range_idx : int
            If multiple scaling ranges were used in fitting, indicates the
            index to use.
        **kwargs : dict
            Optional arguments sent to
            :func:`pymultifracs.viz.plot_cumulants`.[Tuple[int]]
        List of pairs of (j1, j2) ranges of scales for the analysis
    Dq : ndarray, shape (n_exponents, n_rep)
        Fractal dimensions : :math:`D(q)`, y-axis of the multifractal spectrum
    hq : ndarray, shape (n_exponents, n_rep)
        Hölder exponents : :math:`h(q)`, x-axis of the multifractal spectrum
    weighted : str | None
        Weighting mode for the linear regressions. Defaults to None, which is
        no weighting. Possible values are 'Nj' which weighs by number of
        coefficients, and 'bootstrap' which weights by bootstrap-derived
        estimates of variance.
    weights : ndarray
        Weights of the linear regression.
    bootstrapped_obj : MFSpectrum | None
        Storing the bootstrapped version of the structure function if
        bootstraping has been used.

    References
    ----------
    .. [1]  H. Wendt (2008). Contributions of Wavelet Leaders and Bootstrap to
        Multifractal Analysis: Images, Estimation Performance, Dependence
        Structure and Vanishing Moments. Confidence Intervals and Hypothesis
        Tests. Ph.D thesis, Laboratoire de Physique, Ecole Normale Superieure
        de Lyon.
        https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf

    """
    q: np.array
    Dq: np.array = field(init=False)
    hq: np.array = field(init=False)
    U: np.array = field(init=False)
    V: np.array = field(init=False)

    def __post_init__(self, idx_reject, mrq, min_j):

        super().__post_init__(idx_reject, mrq, min_j)

        self.dims = (
            Dim.q, Dim.j, Dim.scaling_range, *mrq.dims[1:])

        self.U = xr.DataArray(
            np.zeros((len(self.q), len(self.j), len(self.scaling_ranges),
                      *mrq.values[self.j.min()].shape[1:])),
            dims=self.dims, coords={Dim.j: self.j, Dim.q: self.q})
        self.V = xr.zeros_like(self.U)

        if self.bootstrapped_obj is not None:
            self.bootstrapped_obj = self.bootstrapped_obj.spectrum

        self._compute(mrq, idx_reject)
        self._compute_fit('U', 'Dq')
        self._compute_fit('V', 'hq')

        self.Dq += 1

    def _compute(self, mrq, idx_reject):
        """
        Computes the multifractal spectrum (Dq, hq)
        """

        # 1. Compute U(j,q) and V(j, q)

        # shape (n_q, n_scales, n_rep)

        for ind_j, j in enumerate(self.j):

            # nj = mrq.nj[j]
            mrq_values_j = mrq.get_values(j, idx_reject)
            dims = mrq_values_j.dims
            mrq_values_j = np.abs(mrq_values_j.values)

            # if 'scaling_range' not in dim_names:
            #     dim_names.insert(1, 'scaling_range')
            #     mrq_values_j = mrq_values_j[:, None]

            # if self.formalism == 'wavelet p-leader':
            #     mrq_values_j = mrq_values_j * mrq.ZPJCorr[None, :, :, ind_j]

            # mrq_values_j = mask_reject(
            #     mrq_values_j, idx_reject, j, mrq.interval_size)

            # idx_nan = np.isnan(mrq_values_j)
            mask_nan = np.isnan(mrq_values_j) | np.isinf(mrq_values_j)
            temp = np.stack(
                [fast_power(mrq_values_j, q) for q in self.q], axis=0)
            # np.nan ** 0 = 1.0, adressed here
            temp[:, mask_nan] = np.nan

            dims = (Dim.q, *dims)

            Z = np.expand_dims(np.nansum(temp, axis=dims.index(Dim.k_j)),
                               axis=dims.index(Dim.k_j))
            Z[Z == 0] = np.nan

            R_j = temp / Z

            # R_j = xr.DataArray(R_j, dims=(Dim.q, *mrq_values_j.dims))

            # reference = [d for d in self.dims if d != Dim.j]
            # mrq_values_j, R_j = _expand_align(
            #     mrq_values_j, R_j, reference_order=reference)

            # nj = ((~mask_nan).sum(axis=0))[None, :]
            N_useful = xr.DataArray(
                (~mask_nan).sum(axis=dims[1:].index(Dim.k_j)),
                dims=[d for d in dims[1:] if d != Dim.k_j]
            )

            new_dims = [d for d in dims if d != Dim.k_j]

            self.V.loc[{Dim.j: j}] = xr.DataArray(fixednansum(
                R_j * np.log2(temp),
                axis=dims.index(Dim.k_j)), dims=new_dims)
            self.U.loc[{Dim.j: j}] = np.log2(N_useful) + xr.DataArray(fixednansum(
                (R_j * np.log2(R_j)),
                axis=dims.index(Dim.k_j)), dims=new_dims)

            idx_unreliable = N_useful < 3

            if idx_unreliable.any():
                self.V.values[:, ind_j, ..., idx_unreliable] = np.nan
                self.U.values[:, ind_j, ..., idx_unreliable] = np.nan

        # self.U.where(np.isinf(self.U.values), np.nan)
        # self.V.where(np.isinf(self.V.values), np.nan)

        self.U.values[np.isinf(self.U)] = np.nan
        self.V.values[np.isinf(self.V)] = np.nan

        # x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
        #     self.scaling_ranges, self.j
        # )

    def V_q(self, q):
        """
        Returns :math:`V_q(j)` for given ``q``.
        """
        return self.V.sel(q=q, method='nearest', tolerance=.1)

    def U_q(self, q):
        """
        Returns :math:`U_q(j)` for given ``q``.
        """
        return self.U.sel(q=q, method='nearest', tolerance=.1)

    def D_q(self):
        """
        Returns :math:`\\mathcal{L}(q)`.
        """
        return self.Dq#.sel(q=q, method='nearest', tolerance=.1)

    def h_q(self):
        """
        Returns :math:`h(q)`.
        """
        return self.hq#.sel(q=q, method='nearest', tolerance=.1)

    def plot(self, filename=None, ax=None, fmt='ko-', range_idx=0,
             signal_idx=0, shift_gamint=False, **plot_kwargs):
        """
        Plot the multifractal spectrum.

        Parameters
        ----------

        filename : str | None
            If not None, saves the figure to ``filename``.
        ax : Axes | None
            Axes where to plot the spectrum. Defaults to None,
            which creates a new figure.
        fmt : str
            Format string for the plot.
        range_idx : int
            If multiple scaling ranges were used in fitting, indicates the
            index to use.
        signal_idx : int
            If using a multivariate signal, index of the signal to plot.
        shift_gamint : bool
            If fractional integration was used, shifts the spectrum on the
            x-axis by :math:`-\\gamma`.
        **plot_kwargs : dict
            Optional arguments sent to the plotting function :func:`plt.plot`.

        """

        ax = plt.gca() if ax is None else ax

        if self.bootstrapped_obj is not None:

            CI_Dq = self.CIE_D_q()
            CI_hq = self.CIE_h_q()

            CI_Dq -= self.D_q()
            CI_hq -= self.h_q()

            CI_Dq = CI_Dq[:, range_idx, signal_idx]
            CI_hq = CI_hq[:, range_idx, signal_idx]

            CI_Dq[:, 1] *= -1
            CI_hq[:, 1] *= -1

            CI_Dq[(CI_Dq < 0) & (CI_Dq > -1e-12)] = 0
            CI_hq[(CI_hq < 0) & (CI_hq > -1e-12)] = 0

            assert (CI_Dq < 0).sum() == 0
            assert (CI_hq < 0).sum() == 0

            CI_Dq = CI_Dq.transpose()
            CI_hq = CI_hq.transpose()

        else:
            CI_Dq, CI_hq = None, None

        shift = 0 if not shift_gamint else self.gamint

        ax.errorbar(self.hq[:, range_idx, signal_idx] - shift,
                    self.Dq[:, range_idx, signal_idx],
                    CI_Dq, CI_hq, fmt,
                    **plot_kwargs)

        ax.set(xlabel=f'Regularity $h{self.regularity_suffix}$',
               ylabel=rf'Fractal dimension $\mathcal{{L}}'
                      f'{self.variable_suffix}(h)$',
               ylim=(0, 1.1), xlim=(0, 1.5),
               title=self.formalism + ' - multifractal spectrum')

        # plt.suptitle()
        plt.draw()

        if filename is not None:
            plt.savefig(filename)
