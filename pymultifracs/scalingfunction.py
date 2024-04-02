from typing import Any
from dataclasses import dataclass, field, InitVar
import inspect

import numpy as np
from scipy import special

import matplotlib.pyplot as plt

# from .multiresquantity import WaveletDec
from .regression import prepare_weights, prepare_regression, \
    linear_regression, compute_R2
from .autorange import compute_Lambda, compute_R, find_max_lambda
from .utils import fast_power, mask_reject, isclose, fixednansum
from . import multiresquantity, viz, robust

@dataclass
class ScalingFunction:
    mrq: InitVar[multiresquantity.WaveletDec]
    scaling_ranges: list[tuple[int]]
    idx_reject: InitVar[dict[int, np.ndarray] | None] = None
    weighted: str | None = None
    bootstrapped_sf: Any | None = None
    formalism: str = field(init=False)
    gamint: float = field(init=False)
    n_sig: int = field(init=False)
    values: np.array = field(init=False)
    slope: np.array = field(init=False)
    intercept: np.array = field(init=False)
    weights: np.ndarray = field(init=False)
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

    def std_values(self):

        from .bootstrap import get_std

        self._check_enough_rep_bootstrap()

        return get_std(self, 'values')
    
    def compute_R2(self):
        return compute_R2(self.values, self.slope, self.intercept, self.weights,
                          [self._get_j_min_max()], self.j)

    def compute_R(self):

        values = self.values.reshape(
            *self.values.shape[:2], self.n_sig, -1)
        slope = self.slope.reshape(*self.slope.shape[:2], self.n_sig, -1)
        intercept = self.intercept.reshape(
            *self.intercept.shape[:2], self.n_sig, -1)

        return compute_R(values, slope, intercept, self.weights,
                         [self._get_j_min_max()], self.j)

    def compute_Lambda(self):

        R = self.compute_R()
        R_b = self.bootstrapped_sf.compute_R()

        return compute_Lambda(R, R_b)

    def find_best_range(self):
        return find_max_lambda(self.compute_Lambda())

    def get_jrange(self, j1=None, j2=None, bootstrap=False):

        if self.bootstrapped_sf is not None and bootstrap:
            if j1 is None:
                j1 = self.bootstrapped_sf.j.min()
            if j2 is None:
                j2 = self.bootstrapped_sf.j.max()

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

        slope = np.zeros(
            (values.shape[0], len(self.scaling_ranges),
             values.shape[-1]))

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j)
        
        self.intercept = np.zeros_like(slope)

        y = values[:, j_min_idx:j_max_idx, :, :]
        
        if self.weighted == 'bootstrap':

            if self.bootstrapped_sf is None:
                std = self.std_values()[:, j_min_idx:j_max_idx]
        
            else:

                if j_min < self.bootstrapped_sf.j.min():
                    raise ValueError(
                        f"Bootstrap minimum scale "
                        f"{self.bootstrapped_sf.j.min()} inferior to minimum "
                        f"scale {j_min} used in estimation")
                
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

        # self.weights *= nan_weighting
        
        slope, self.intercept = linear_regression(x, y, self.weights)

        if out_name is not None:
            slope = setattr(self, out_name, slope)
        else:
            self.slope = slope

    def _check_enough_rep_bootstrap(self):

        if (ratio := self.n_rep // self.n_sig) < 2:
            raise ValueError(
                f'n_rep = {ratio} per original signal too small to build '
                'confidence intervals'
                )

    def _get_bootstrapped_sf(self):

        if self.bootstrapped_sf is None:
            bootstrapped_sf = self
        else:
            bootstrapped_sf = self.bootstrapped_sf

        bootstrapped_sf._check_enough_rep_bootstrap()

        return bootstrapped_sf
    
    def _check_bootstrap_sf(self):

        if self.bootstrapped_sf is None:
            raise ValueError(
                "Bootstrapped mrq needs to be computed prior to estimating "
                "empirical estimators")

        self.bootstrapped_sf._check_enough_rep_bootstrap()

    def __getattr__(self, name):

        if name[:3] == 'CI_':
            from .bootstrap import get_confidence_interval

            bootstrapped_sf = self._get_bootstrapped_sf()

            return get_confidence_interval(bootstrapped_sf, name[3:])

        elif name[:4] == 'CIE_':
            from .bootstrap import get_empirical_CI

            self._check_bootstrap_sf()

            return get_empirical_CI(self.bootstrapped_sf, self, name[4:])

        elif name[:3] == 'VE_':
            from .bootstrap import get_empirical_variance

            self._check_bootstrap_sf()

            return get_empirical_variance(self.bootstrapped_sf, self,
                                          name[3:])

        elif name[:3] == 'SE_':

            from .bootstrap import get_empirical_variance

            self._check_bootstrap_sf()

            return np.sqrt(
                get_empirical_variance(self.bootstrapped_sf, self,
                                       name[3:](self)))

        elif name[:2] == 'V_':

            from .bootstrap import get_variance

            bootstrapped_sf = self._get_bootstrapped_sf()

            return get_variance(bootstrapped_sf, name[2:])

        elif name[:4] == 'STD_':

            from .bootstrap import get_std

            bootstrapped_sf = self._get_bootstrapped_sf()

            return get_std(bootstrapped_sf, name[4:])

        elif name == 'n_rep':
            return self.intercept.shape[-1]

        return self.__getattribute__(name)


@dataclass(kw_only=True)
class StructureFunction(ScalingFunction):
    q: np.array
    H: np.array = field(init=False)

    def __post_init__(self, mrq, idx_reject):

        self.gamint = mrq.gamint
        self.n_sig = mrq.n_sig
        self.formalism = mrq.get_formalism()
        self.j = np.array(list(mrq.values))

        if self.bootstrapped_sf is not None:
            self.bootstrapped_sf = self.bootstrapped_sf.structure

        self._compute(mrq, idx_reject)

        self.slope = np.zeros(
            (len(self.q), len(self.scaling_ranges), mrq.n_rep))
        self.intercept = np.zeros_like(self.slope)

        self._compute_fit()

    def _compute(self, mrq, idx_reject):

        self.values = np.zeros(
            (len(self.q), len(self.j), len(self.scaling_ranges), mrq.n_rep))

        for ind_j, j in enumerate(self.j):

            c_j = mrq.get_values(j, idx_reject)

            for ind_q, q in enumerate(self.q):
                self.values[ind_q, ind_j, :] = np.log2(
                    np.nanmean(fast_power(np.abs(c_j), q), axis=0))

        self.values[np.isinf(self.values)] = np.nan

    def _get_H(self):
        return (self.slope[self.q == 2][0] / 2)

    def S_q(self, q):

        out = self.values[isclose(q, self.q)][0]
        out = out.reshape(out.shape[0], len(self.scaling_ranges), self.n_sig, -1)

        return out

    def s_q(self, q):

        out = self.slope[isclose(q, self.q)][0]
        out = out.reshape(out.shape[0], self.n_sig, -1)

        return out

    def __getattr__(self, name):

        if name == 'H':
            return self._get_H()

        if name == 'S2':
            out = self.values[self.q == 2]
            return out.reshape(out.shape[0], self.n_sig, -1)
        
        if name == 'zeta':
            return self.slope

        if (super_attr := super().__getattr__(name)) is not None:
            return super_attr

        return self.__getattribute__(name)

    def plot(self, figlabel='Structure Functions', nrow=4, filename=None,
             ignore_q0=True, figsize=None, scaling_range=0, plot_scales=None,
             plot_CI=True, signal_idx=0):
        """
        Plots the structure functions.
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
                                 num=figlabel,
                                 squeeze=False,
                                 figsize=figsize,
                                 sharex=True)

        # fig.suptitle(self.formalism +
        #              r' - structure functions $\log_2(S(j,q))$')

        x = self.j[idx]

        counter = 0

        for ind_q, q in enumerate(self.q):

            if q == 0.0 and ignore_q0:
                continue

            y = self.S_q(q)[idx, scaling_range, signal_idx, 0]

            if self.bootstrapped_sf is not None and plot_CI:

                _, _, j_min_CI, j_max_CI = self.bootstrapped_sf.get_jrange(
                    j1, j2)

                CI = self.CIE_S_q(q)[j_min_CI:j_max_CI, scaling_range, signal_idx]

                CI -= y[:, None]
                CI[:, 1] *= -1
                assert (CI < 0).sum() == 0
                CI = CI.transpose()

            else:
                CI = None

            ax = axes[counter % nrow][counter // nrow]
            ax.errorbar(x, y, CI, fmt='r--.', zorder=4)
            ax.set_xlabel('j')
            ax.set_ylabel(f'q = {q:.3f}')
            ax.tick_params(bottom=False, top=False, which='minor')

            counter += 1

            x0, x1 = self.scaling_ranges[scaling_range]
            slope = self.slope[ind_q, scaling_range, 0]
            intercept = self.intercept[ind_q, scaling_range, 0]

            assert x0 in x, "Scaling range not included in plotting range"
            assert x1 in x, "Scaling range not included in plotting range"

            y0 = slope*x0 + intercept
            y1 = slope*x1 + intercept

            if self.bootstrapped_sf is not None and plot_CI:
                CI = self.CIE_s_q(q)[scaling_range, signal_idx]
                CI_legend = f"; [{CI[0]:.1f}, {CI[1]:.1f}]"
            else:
                CI_legend = ""

            legend = rf'$s_{{{q:.2f}}}$ = {slope:.2f}' + CI_legend

            ax.plot([x0, x1], [y0, y1], color='k',
                    linestyle='-', linewidth=2, label=legend, zorder=5)
            ax.legend()

        for j in range(counter, len(axes.flat)):
            fig.delaxes(axes[j % nrow][j // nrow])

        # plt.draw()

        if filename is not None:
            plt.savefig(filename)

    def plot_scaling(self, figlabel='Scaling Function', filename=None,
                    ax=None, signal_idx=0, range_idx=0, **plot_kw):
        
        assert len(self.q) > 1, ("This plot is only possible if more than 1 q",
                                 " value is used")

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.q, self.slope[:, range_idx, signal_idx], **plot_kw)
        ax.set(
            xlabel = 'q', ylabel=r'$\zeta(q)$',
            title=self.formalism + ' - scaling function')

        # plt.draw()

        if filename is not None:
            plt.savefig(filename)


@dataclass(kw_only=True)
class Cumulants(ScalingFunction):
    r"""
    Computes and analyzes cumulant.
    
    .. note:: Should not be initialized directly but instead computed from `mf_analysis`.

    Attributes
    ----------
    n_cumul : int
        Number of computed cumulants.
    scaling_ranges : List[Tuple[int]]
        List of pairs of scales delimiting the temporal scale support over
        which the estimates are regressed.
    weighted : str | None
        Whether weighted regression was performed.
    robust_kwargs : Dict[str, object]:
        Arguments used in robust estimation.
    m : ndarray, shape (n_cumulants,)
        List of the m values (cumulants), in order presented in the value
        arrays.
    j : ndarray, shape (n_scales,)
        List of the j values (scales), in order presented in the value arrays.
    values : ndarray, shape (n_cumulants, n_scales, n_rep)
        :math:`C_m(j)`.
    log_cumulants : ndarray, shape (n_cumulants, n_rep)
        :math:`(c_m)_m`, slopes of the curves :math:`j \times C_m(j)`.
    var_log_cumulants : ndarray, shape (n_cumulants, n_rep)
        Estimates of the variance of log-cumulants.
        .. warning:: var_log_cumulants was not debugged
    formalism : str
        Formalism used. Can be any of 'wavelet coefs', 'wavelet leaders',
        or 'wavelet p-leaders'.
    gamint : float
        Value of gamint used in the computation of the underlying MRQ.
    wt_name : str
        Name of the wavelet used in the underlying MRQ.
    nj : dict(ndarray)
        Number of coefficients at scale j
        Arrays are of the shape (n_rep,).

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

    def __post_init__(self, mrq, idx_reject, robust, robust_kwargs):

        self.formalism = mrq.get_formalism()
        self.n_sig = mrq.n_sig
        self.gamint = mrq.gamint
        self.j = np.array(list(mrq.values))

        if self.bootstrapped_sf is not None:
            self.bootstrapped_sf = self.bootstrapped_sf.cumulants

        self.m = np.arange(1, self.n_cumul+1)
        
        self.values = np.zeros(
            (len(self.m), len(self.j), len(self.scaling_ranges), mrq.n_rep))

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
 
    def _compute_robust(self, mrq, idx_reject, **robust_kwargs):

        moments = np.zeros_like(self.values)
        aux = np.zeros_like(moments)

        for ind_j, j in enumerate(self.j):

            T_X_j = np.abs(mrq.values[j])
            T_X_j = T_X_j[:, None, :]

            if self.formalism == 'wavelet p-leader':
                T_X_j = T_X_j * mrq.ZPJCorr[None, :, :, ind_j]

            log_T_X_j = np.log(T_X_j)

            # dropping infinite coefsx
            log_T_X_j[np.isinf(log_T_X_j)] = np.nan

            log_T_X_j = mask_reject(
                log_T_X_j, idx_reject, j, mrq.interval_size)

            values = robust.compute_robust_cumulants(
                log_T_X_j, self.m, **self.robust_kwargs)

            self.values[:, ind_j] = values

    def _compute(self, mrq, idx_reject):

        moments = np.zeros_like(self.values)

        for ind_j, j in enumerate(self.j):

            T_X_j = np.abs(mrq.get_values(j, None))

            np.log(T_X_j, out=T_X_j)

            mask_nan = np.isnan(T_X_j)
            mask_nan |= np.isinf(T_X_j)
            
            if idx_reject is not None:
                delta = (mrq.interval_size - 1) // 2
                mask_nan |= idx_reject[j][delta:-delta]

            T_X_j[mask_nan] = 0

            N_useful = (~mask_nan).sum(axis=0)

            for ind_m, m in enumerate(self.m):

                moments[ind_m, ind_j] = np.sum(fast_power(T_X_j, m), axis=0)
                np.divide(
                    moments[ind_m, ind_j], N_useful, out=moments[ind_m, ind_j])

                idx_unreliable = N_useful < 3

                for i in range(idx_unreliable.shape[0]):
                    moments[ind_m, ind_j, i, idx_unreliable[i]] = np.nan

                if m == 1:
                    self.values[ind_m, ind_j] = moments[ind_m, ind_j]
                else:
                    aux = 0

                    for ind_n, n in enumerate(np.arange(1, m)):
                        aux += (special.binom(m-1, n-1)
                                * self.values[ind_n, ind_j]
                                * moments[ind_m-ind_n-1, ind_j])

                    self.values[ind_m, ind_j] = moments[ind_m, ind_j] - aux

    def __getattr__(self, name):

        if name[0] == 'c' and len(name) == 2 and name[1:].isdigit():

            out = self.log_cumulants[self.m == int(name[1])][0]
            out = out.reshape(out.shape[0], self.n_sig, -1)

            return out

        if name[0] == 'C' and len(name) == 2 and name[1:].isdigit():

            out = self.values[self.m == int(name[1])][0]
            out = out.reshape(out.shape[0], out.shape[1], self.n_sig, -1)

            return out

        if name == 'M':
            return -self.c2
        
        if (super_attr := super().__getattr__(name)) is not None:
            return super_attr

        return self.__getattribute__(name)

    def plot(self, figsize=None, nrow=3, j1=None, filename=None,
             scaling_range=0, n_cumul=None, signal_idx=0, **kwargs):

        return viz.plot_cumulants(
            self, figsize, nrow, j1, filename, scaling_range,
            n_cumul=n_cumul, signal_idx=signal_idx, **kwargs)


@dataclass(kw_only=True)
class MFSpectrum(ScalingFunction):
    """
    Estimates the Multifractal Spectrum

    Based on equations 2.74 - 2.78 of Herwig Wendt's thesis [1]_

    Parameters
    ----------
    mrq : MultiResolutionQuantity
        Multi resolution quantity to analyze.
    scaling_ranges: List[Tuple[int]]
        List of pairs of (j1, j2) ranges of scales for the analysis
    q : ndarray, shape (n_exponents,)
        Exponents used construct the multifractal spectrum
    boostrapped_mfa: MFractalVar | None
        Output of the MFA of bootstrapped MRQs.
    weighted : str | None
        Whether to used weighted linear regressions.

    Attributes
    ----------
    formalism : str
        Formalism used. Can be any of 'wavelet coefs', 'wavelet leaders',
        or 'wavelet p-leaders'.
    j : ndarray, shape (n_scales,)
        List of the j values (scales), in order presented in the value arrays.
    scaling_ranges: List[Tuple[int]]
        List of pairs of (j1, j2) ranges of scales for the analysis
    weighted : str | None
        If not None, indicates the weighting approach used for regression
    q : ndarray, shape(n_exponents,)
        Exponents used construct the multifractal spectrum
    Dq : ndarray, shape (n_exponents, n_rep)
        Fractal dimensions : :math:`D(q)`, y-axis of the multifractal spectrum
    hq : ndarray, shape (n_exponents, n_rep)
        Hölder exponents : :math:`h(q)`, x-axis of the multifractal spectrum
    U : ndarray, shape (n_scales, n_exponents, n_rep)
        :math:`U(j, q)`
    V : ndarray, shape (n_scales, n_exponents, n_rep)
        :math:`V(j, q)`

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

    def __post_init__(self, mrq, idx_reject):

        self.formalism = mrq.get_formalism()
        self.gamint = mrq.gamint
        self.n_sig = mrq.n_sig
        self.j = np.array(list(mrq.values))

        self.U = np.zeros(
            (len(self.q), len(self.j), len(self.scaling_ranges), mrq.n_rep))
        self.V = np.zeros_like(self.U)

        if self.bootstrapped_sf is not None:
            self.bootstrapped_sf = self.bootstrapped_sf.spectrum

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
            mrq_values_j = np.abs(mrq.get_values(j, idx_reject))

            # if self.formalism == 'wavelet p-leader':
            #     mrq_values_j = mrq_values_j * mrq.ZPJCorr[None, :, :, ind_j]

            # mrq_values_j = mask_reject(
            #     mrq_values_j, idx_reject, j, mrq.interval_size)

            idx_nan = np.isnan(mrq_values_j)
            temp = np.stack([fast_power(mrq_values_j, q) for q in self.q],
                            axis=0)
            # np.nan ** 0 = 1.0, adressed here
            temp[:, idx_nan] = np.nan

            Z = np.nansum(temp, axis=1)[:, None, :]
            Z[Z == 0] = np.nan
            R_j = temp / Z
            
            nj = ((~idx_nan).sum(axis=0))[None, :]
            self.V[:, ind_j] = fixednansum(R_j * np.log2(mrq_values_j), axis=1)
            self.U[:, ind_j] = np.log2(nj) + fixednansum((R_j * np.log2(R_j)),
                                                         axis=1)

        self.U[np.isinf(self.U)] = np.nan
        self.V[np.isinf(self.V)] = np.nan

        x, n_ranges, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
            self.scaling_ranges, self.j
        )

    def V_q(self, q):
        out = self.V[isclose(q, self.q)][0]
        return out.reshape(out.shape[0], self.n_sig, -1)

    def U_q(self, q):
        out = self.U[np.isclose(q, self.q)][0]
        return out.reshape(out.shape[0], self.n_sig, -1)
    
    def D_q(self):
        return self.Dq.reshape(
            len(self.q), len(self.scaling_ranges), self.n_sig, -1)#[
            #     :
            # ]
    
    def h_q(self):
        return self.hq.reshape(
            len(self.q), len(self.scaling_ranges), self.n_sig, -1)#[
            #     :
            # ]

    def plot(self, figlabel='Multifractal Spectrum', filename=None, ax=None,
             fmt='ko-', scaling_range=0, signal_idx=0, shift_gamint=False,
             **plot_kwargs):
        """
        Plot the multifractal spectrum.

        Parameters
        ----------
        figlabel : str
            Figure title
        filename : str | None
            If not None, path used to save the figure
        """

        ax = plt.gca() if ax is None else ax

        if self.bootstrapped_sf is not None:

            CI_Dq = self.CIE_D_q()
            CI_hq = self.CIE_h_q()

            CI_Dq -= self.D_q()
            CI_hq -= self.h_q()

            CI_Dq = CI_Dq[:, scaling_range, signal_idx]
            CI_hq = CI_hq[:, scaling_range, signal_idx]

            CI_Dq[:, 1] *= -1
            CI_hq[:, 1] *= -1

            CI_Dq[(CI_Dq < 0) & (CI_Dq > -1e-12)] = 0
            CI_hq[(CI_hq < 0) & (CI_hq > -1e-12)] = 0

            assert(CI_Dq < 0).sum() == 0
            assert(CI_hq < 0).sum() == 0

            CI_Dq = CI_Dq.transpose()
            CI_hq = CI_hq.transpose()

        else:
            CI_Dq, CI_hq = None, None

        shift = 0 if not shift_gamint else self.gamint

        ax.errorbar(self.hq[:, scaling_range, signal_idx] - shift,
                    self.Dq[:, scaling_range, signal_idx],
                    CI_Dq, CI_hq, fmt,
                    **plot_kwargs)

        ax.set(xlabel='Regularity $h$', ylabel='Fractal dimension $D(h)$',
               ylim=(0, 1.1), xlim=(0, 1.5),
               title=self.formalism + ' - multifractal spectrum')

        # plt.suptitle()
        plt.draw()

        if filename is not None:
            plt.savefig(filename)
