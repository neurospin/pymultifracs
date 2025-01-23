"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import dataclass, field
import inspect
import warnings
from typing import Any

import numpy as np
import pywt

from .utils import get_filter_length, max_scale_bootstrap, _correct_pleaders,\
    mask_reject, AbstractDataclass, max_scale_bootstrap
from . import viz, wavelet, estimation

@dataclass(kw_only=True)
class MultiResolutionQuantityBase(AbstractDataclass):
    n_sig: int
    bootstrapped_obj: Any | None = None
    origin_mrq: Any | None = None

    # def get_nj(self):
    #     """
    #     Returns nj as a list
    #     """
    #     return list(self.nj.values())

    # def update_nj(self):
    #     self.nj = {
    #         scale: (~np.isnan(self.values[scale])).sum(axis=0)
    #         for scale in self.values
    #     }

    def _from_dict(self, d):
        r"""
        Method to instanciate a dataclass by passing a dictionary with
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

        cls = type(self)

        parameters = {
            name: getattr(self, name)
            for name in inspect.signature(cls).parameters.keys()
        }

        input = parameters.copy()
        input.update(d)

        return cls(**{
            k: v for k, v in input.items()
            if k in parameters
        })

    def sup_coeffs(self, n_ranges, j_max, j_min, scaling_ranges, idx_reject):

        sup_coeffs = np.ones((j_max - j_min + 1, n_ranges, self.n_rep))

        for i, (j1, j2) in enumerate(scaling_ranges):
            for j in range(j1, j2 + 1):

                # c_j = np.abs(self.values[j])[:, None, :]

                c_j = np.abs(self.get_values(j, idx_reject))

                # c_j = mask_reject(c_j, idx_reject, j, 1)
                
                sup_c_j = np.nanmax(c_j, axis=0)
                sup_coeffs[j-j_min, i] = sup_c_j

        return sup_coeffs

    def j2_eff(self):
        return max(list(self.values))


@dataclass(kw_only=True)
class WaveletDec(MultiResolutionQuantityBase):
    r"""
    Wavelet Coefficient Decomposition.

    It represents the wavelet coefficients of a signal :math:`d_X(j, k)`

    .. note:: Should not be instantiated directly but instead created using the `wavelet_analysis` 
        function.

    Attributes
    ----------
    values : dict[int, ndarray]
        ``values[j]`` contains the coefficients at the scale j.
        Arrays are of the shape (nj, n_rep)
    n_sig : int
        Number of underlying signals in the wavelet decomposition. May not
        match the dimensionality of the values arrays in case there are
        multiple repetitions associated to a single signal.
    gamint : float
        Fractional integration used in the computation of the MRQ.
    wt_name : str
        Name of the wavelet used for the decomposition.
    interval_size : int
        Width of the coef interval over which the MRQ was computed.
    origin_mrq : WaveletDec | None
        If MRQ is derived from another mrq, refers to the original MRQ.
    bootstrapped_obj : WaveletDec | None
        Storing the bootstrapped version of the MRQ if bootstraping has been
        used.
    """
    wt_name: str
    gamint: float = 0
    values: dict = field(default_factory=dict)
    origin_mrq: MultiResolutionQuantityBase | None = None
    interval_size: int = field(init=False, default=1)

    def get_nj_interv(self, j1=None, j2=None):
        """
        Returns nj as an array, for j in [j1,j2]
        """

        if j1 is None:
            j1 = min(self.values)
        if j2 is None:
            j2 = max(self.values)

        return np.array([(~np.isnan(self.values[j])).sum(axis=0)
                         for j in range(j1, j2+1)])

    def bootstrap(self, R, min_scale=1, idx_reject=None):
        r"""
        Bootstrap the multi-resolution quantity by repeating.

        Parameters
        ----------
        R : int
            Number of repetitions of the bootstrap.
        min_scale : int
            Minimum scale that will be kept in the bootstrapped MRQ, used to
            save memory space when analyzing the coarse scales of long time
            series.
        idx_reject : dict[str, np.ndarray] | None
            Dictionary of rejected values, by default None which means no
            rejected values.

        Returns
        -------
        WaveletDec
            MRQ containing the bootstrapped values.
        """

        from .bootstrap import circular_leader_bootstrap

        block_length = get_filter_length(self.wt_name)
        max_scale = max_scale_bootstrap(self, idx_reject)

        self.bootstrapped_obj = circular_leader_bootstrap(
            self, min_scale, max_scale, block_length, R)

        # j = np.array([*self.values])
        #
        # if min_scale > j.min():
        #     self.values = {scale: value
        #                    for scale, value in self.values.items()
        #                    if scale >= min_scale}
        #     self.nj = {scale: nj for scale, nj in self.nj.items()
        #                if scale >= min_scale}

        return self.bootstrapped_obj
    
    def scale2freq(self, scale, sfreq):
        """
        Get the frequencies associated to scales.

        Parameters
        ----------
        scale : int | float | ndarray
            Scales to convert to frequency.
        sfreq : float
            Sampling frequency of the signal.

        Returns
        -------
        freq : float | ndarray
            Frequencies associated to `scales`.
        """
        return pywt.scale2frequency(self.wt_name, 2 ** scale) * sfreq
        
    def freq2scale(self, freq, sfreq):
        """
        Get the scales associated to frequencies.

        Parameters
        ----------
        freq : float | ndarray
            Frequencies to convert to scales.
        sfreq : float
            Sampling frequency of the signal.

        Returns
        -------
        scales : float | ndarray
            Scales associated to `freq`.
        """
        return np.log2(pywt.frequency2scale(self.wt_name, freq / sfreq))
    
    def max_scale_bootstrap(self, idx_reject=None):
        """
        Maximum scale at which bootstrapping can be done

        Parameters
        ----------
        idx_reject : dict[str, np.ndarray] | None
            Dictionary of rejected values, by default None which means no
            rejected values.

        Returns
        -------
        Scale : int
        """
        return max_scale_bootstrap(self, idx_reject)

    @classmethod
    def bootstrap_multiple(cls, R, min_scale, mrq_list):

        from .bootstrap import circular_leader_bootstrap

        block_length = max([
            get_filter_length(mrq.wt_name) for mrq in mrq_list
        ])

        max_scale = min([
            max_scale_bootstrap(mrq) for mrq in mrq_list
        ])

        # j2_eff = np.array([mrq.j2_eff() for mrq in mrq_list])
        # wrong_idx = max_scale < j2_eff

        # if wrong_idx.any():
        #     raise ValueError(f'Maximum bootstrapping scale {max_scale} is '
        #                      f'inferior to the j2 chosen when computing '
        #                      f'wavelet leaders for indices {wrong_idx}.')

        return circular_leader_bootstrap(mrq_list, min_scale, max_scale,
                                         block_length, R)

    def add_values(self, coeffs, j):
        self.values[j] = coeffs

    def get_values(self, j, idx_reject=None, reshape=False):

        # Case where bootstrapping was done
        if self.values[j].ndim == 3:
            out = self.values[j]
        else:
            out = self.values[j][:, None, :]
        
        # Bootstrapped mrq needs to realign into signal and repetitions
        if reshape: # and self.n_rep != self.n_sig
            out = out.reshape(self.values[j].shape[0], 1, self.n_sig, -1)

        if idx_reject is None:
            return out

        return mask_reject(
            out, idx_reject, j, self.interval_size)

    def plot(self, j1, j2, ax=None, vmin=None, vmax=None, cbar=True,
             figsize=(4.5, 1.5), gamma=.3, nan_idx=None, signal_idx=0,
             cbar_kw=None, cmap='magma'):
        """
        Plot the multi-resolution quantity.

        Parameters
        ----------
        j1 : int
            Initial scale from which to display the values.
        j2 : int
            Maximal scale from which to display the values.
        ax : matplotlib.pyplot.axes | None
            pyplot axes, defaults to None which creates a new figure.
        vmin : float | None
            Minimal value of the colorbar, by default None which uses the
            minimal value in the data.
        vmax : float | None
            Maximal value of the colorbar, by default None which uses the
            maximal value in the data.
        cbar : bool
            Whether to display a colorbar.
        figsize : tuple, optional
            Size of the figure, used if ax is None.
        gamma : float, optional
            Exponent of the power-law color normalization, set to 1 for no
            normalization.
        nan_idx : dict[str, ndarray] | None
            Index of values to highlight.
        signal_idx : int, optional
            Index of the signal to plot, defaults to the first signal.
        cbar_kw : dict | None
            Arguments to pass to the colorbar function call
        cmap : str
            Colormap for the plot.
        """
        viz.plot_coef(
            self, j1, j2, ax=ax, vmin=vmin, vmax=vmax, cbar=cbar,
            figsize=figsize, gamma=gamma, nan_idx=nan_idx,
            signal_idx=signal_idx, cbar_kw=cbar_kw, cmap=cmap)

    def get_formalism(self):
        """
        Obtains the fomalism of the multi-resolution quantity

        Returns
        -------
        formalism : str
        """

        return 'wavelet coef'
    
    def integrate(self, gamint):
        """
        Fractionally integrate the wavelet coefficients.

        Parameters
        ----------
        gamint : float
            Fractional integration coefficient

        Returns
        -------
        integrated : WaveletDec
        """
        return wavelet.integrate_wavelet(self, gamint)
    
    def get_leaders(self, p_exp, interval_size=3, gamint=None):
        """
        Compute (p-)leaders from the wavelet coefficients

        Parameters
        ----------
        p_exp : float | np.inf
            np.inf means wavelet leaders will also be computed, and a float
            sets the value of the p exponent implying a wavelet p-leader
            formalism.
        interval_size : int
            Width of the time shift interval over which the leaders are
            computed, by default the usual value of 3.
        gamint : int | None
            Fractional integration coefficient, by default None which means
            that current integration will be conserved.

        Returns
        -------
        WaveletLeader
            Wavelet (p-)leader derived from the coefficients.
        """

        if type(self) != WaveletDec:
            return self.origin_mrq.get_leaders(p_exp, interval_size, gamint)
        
        if gamint is None:
            gamint = self.gamint

        if self.gamint == gamint:
            return wavelet.compute_leaders(self, p_exp,  interval_size)

        if self.origin_mrq is not None:
            return self.origin_mrq.get_leaders(p_exp, interval_size, gamint)
        
        integ = self.integrate(gamint - self.gamint)
        return integ.get_leaders(p_exp, interval_size)

    def get_wse(self, theta=.5, gamint=None):
        """
        Compute weak scaling exponents from the wavelet coefficients

        Parameters
        ----------
        theta : float, optional
            Parameter controlling the angle of the cone over which the weak 
            scaling exponents are computed
        gamint : float, optional
            Fractional integration coefficient, defaults to 0 (no integration).

        Returns
        -------
        WaveletWSE
            Weak scaling exponent derived from the coefficients.
        """

        if type(self) != WaveletDec:
            return self.origin_mrq.get_wse(theta, gamint)

        if gamint is None:
            gamint = self.gamint

        if gamint == self.gamint:
            return wavelet.compute_wse(self, theta)

        if self.origin_mrq is not None:
            return self.origin_mrq.compute_wse(theta, gamint)
        
        integ = self.integrate(gamint - self.gamint)
        return integ.get_wse(theta, gamint)

    def auto_integrate(self, scaling_ranges, weighted=None, idx_reject=None):

        hmin, _ = estimation.estimate_hmin(
            self, scaling_ranges, weighted, idx_reject)

        hmin = hmin.min()

        if hmin // .5 > 0:
            gamint = 0
        else:
            gamint = -.5 * (hmin.min() // .5)

            if gamint + hmin < 0.25:
                gamint += .5

        if gamint != self.gamint:
            return self.integrate(gamint)

        return self

    def check_regularity(self, scaling_ranges, weighted=None, idx_reject=None):
        r"""
        Verify that the MRQ has enough regularity for analysis.

        Parameters
        ----------
        scaling_ranges : list[tuple[int, int]]
            List of pairs of (j1, j2) ranges of scales for the analysis.
        weighted : str | None
            Weighting mode for the linear regressions. Defaults to None, which
            is no weighting. Possible values are 'Nj' which weighs by number of
            coefficients, and 'bootstrap' which weights by bootstrap-derived
            estimates of variance.
        idx_reject : Dict[int, ndarray]
            Dictionary associating each scale to a boolean array indicating
            whether certain coefficients should be removed.

        Returns
        -------
        ndarray
            Estimate of the minimal Hölder exponent in the MRQ.
        """
        
        hmin, _ = estimation.estimate_hmin(
            self, scaling_ranges, weighted, idx_reject)
        
        if hmin.max() <= 0:
            raise ValueError(
                f"Maximum hmin = {hmin.max()} <= 0, no signal can be "
                "analyzed. A larger value of gamint or different scaling range"
                " should be selected.")

        if hmin.min() <= 0:
            warnings.warn(
                f"Minimum hmin = {hmin.min()} <= 0, multifractal analysis "
                "cannot be applied. A larger value of gamint) should be "
                "selected.")
            
        return hmin

    def __getattribute__(self, name: str) -> Any:

        if name == 'filt_len':
            return get_filter_length(self.wt_name)

        if name == 'n_rep':
            if len(self.values) > 0:
                return self.values[[*self.values][0]].shape[-1]

        # if name == 'n_sig' and super().__getattribute__('n_sig') is None:
        #     return 1

        return super().__getattribute__(name)

    # def __getattr__(self, name):
    #     return super().__getattr__(name)


@dataclass(kw_only=True)
class WaveletLeader(WaveletDec):
    """
    Wavelet Leader Representation.

    It contains the wavelet (p-)leader representation of a signal
    :math:`L_X(j, k)`.
    
    .. note:: Should not be instantiated directly but instead created using the :func:`WaveletDec.get_leaders` method.

    Attributes
    ----------
    values : dict[int, ndarray]
        ``values[j]`` contains the coefficients at the scale j.
        Arrays are of the shape (nj, n_rep)
    n_sig: int
        Number of underlying signals in the wavelet decomposition. May not
        match the dimensionality of the values arrays in case there are
        multiple repetitions associated to a single signal.
    gamint : float
        Fractional integration used in the computation of the MRQ.
    wt_name : str
        Name of the wavelet used for the decomposition.
    interval_size : int
        Width of the coef interval over which the MRQ was computed.
    origin_mrq : WaveletDec | None
        If MRQ is derived from another mrq, refers to the original MRQ.
    bootstrapped_obj : :class:`.MultiResolutionQuantity` | None
        Storing the bootstrapped version of the MRQ if bootstraping has been
        used.
    p_exp: float | np.inf
        P exponent used to compute the MRQ. np.inf indicates wavelet leaders,
        float indicates p-leaders.
    eta_p : float | None
        Only for p-leaders, wavelet scaling function :math:`\eta(p)`.
        By default only computed during mf_analysis.
    ZPJCorr : ndarray | None
        Only for p-leaders, correction factor for the finite size effects,
        dependent on `eta_p`.
    """
    p_exp: float
    interval_size: int = 1
    eta_p: np.ndarray = field(init=False, repr=False, default=None)
    ZPJCorr: np.ndarray = field(init=False, default=None)

    def bootstrap(self, R, min_scale=1, idx_reject=None):

        self.bootstrapped_obj, self.origin_mrq.bootstrapped_obj = \
            self.__class__.bootstrap_multiple(
                R, min_scale, [self, self.origin_mrq])

        return self.bootstrapped_obj

        # self.origin_mrq.bootstrap

        # from .bootstrap import circular_leader_bootstrap

        # block_length = get_filter_length(self.wt_name)
        # max_scale = max_scale_bootstrap(self, idx_reject)

        # self.bootstrapped_obj = circular_leader_bootstrap(
        #     self, min_scale, max_scale, block_length, R)

        # # j = np.array([*self.values])
        # #
        # # if min_scale > j.min():
        # #     self.values = {scale: value
        # #                    for scale, value in self.values.items()
        # #                    if scale >= min_scale}
        # #     self.nj = {scale: nj for scale, nj in self.nj.items()
        # #                if scale >= min_scale}

        # return self.bootstrapped_obj

    def get_formalism(self):
        """
        Obtains the fomalism of the multi-resolution quantity

        Returns
        -------
        tr
        """

        if self.p_exp == np.inf:
            return 'wavelet leader'
        return 'wavelet p-leader'
    
    def integrate(self, gamint):
        """
        Re-compute the (p-)leaders on the fractionally integrated wavelet
        coefficients.

        Parameters
        ----------
        gamint : float
            Fractional integration coefficient

        Returns
        -------
        WaveletLeader
        """
        return self.get_leaders(self.p_exp, self.interval_size, gamint)

    def get_values(self, j, idx_reject=None, reshape=False):
        
        # Case where bootstrapping was done
        if self.values[j].ndim == 3:
            return super().get_values(j, idx_reject, reshape)

        if self.p_exp == np.inf or self.eta_p is None:
            return super().get_values(j, idx_reject, reshape)

        if self.ZPJCorr is None:
            self.correct_pleaders()

        ZPJCorr = self.ZPJCorr[None, :, :, j - min(self.values)]

        if reshape:
            ZPJCorr = ZPJCorr[..., None]

        return ZPJCorr * super().get_values(j, idx_reject, reshape)

    def get_leaders(self, p_exp, interval_size=3, gamint=None):

        if (p_exp == self.p_exp
                and interval_size == self.interval_size
                and (gamint == self.gamint or gamint is None)):

            return self

        return super().get_leaders(p_exp, interval_size, gamint)

    def correct_pleaders(self):

        # No correction if infinite p
        if self.p_exp == np.inf:
            return

        self.ZPJCorr = _correct_pleaders(
            self, self.p_exp, min(self.values), max(self.values))
        
        return self.ZPJCorr
    
    def auto_integrate(self, scaling_ranges, weighted=None, idx_reject=None):

        if self.p_exp == np.inf:
            return super().auto_integrate(
                scaling_ranges, weighted, idx_reject).get_leaders(
                    self.p_exp, self.interval_size, self.gamint)

        eta_p = estimation._estimate_eta_p(
            self.origin_mrq, self.p_exp, scaling_ranges, weighted, idx_reject)

        eta_p = eta_p.min()

        if eta_p // .5 > 0:
            gamint = 0
        else:
            gamint = -.5 * (eta_p.min() // .5)

            if gamint + eta_p < 0.25:
                gamint += .5

        if gamint != self.gamint:
            return self.origin_mrq.get_leaders(
                self.p_exp, self.interval_size, gamint)
        
        return self

    def check_regularity(self, scaling_ranges, weighted=None,
                          idx_reject=None):
        """
        Verify that the MRQ has enough regularity for analysis.

        Parameters
        ----------
        scaling_ranges : list[tuple[int, int]]
            List of pairs of (j1, j2) ranges of scales for the analysis.
        weighted : str | None
            Weighting mode for the linear regressions. Defaults to None, which
            is no weighting. Possible values are 'Nj' which weighs by number of
            coefficients, and 'bootstrap' which weights by bootstrap-derived
            estimates of variance.
        idx_reject : Dict[int, ndarray]
            Dictionary associating each scale to a boolean array indicating
            whether certain coefficients should be removed.
        """

        if self.p_exp == np.inf:
            return super().check_regularity(
                scaling_ranges, weighted, idx_reject)

        eta_p = estimation._estimate_eta_p(
            self.origin_mrq, self.p_exp, scaling_ranges, weighted, idx_reject)

        if eta_p.max() <= 0:
            raise ValueError(
                f"Maximum eta(p) = {eta_p.max()} <= 0, no signal can be "
                "analyzed. A smaller value of p (or larger value of gamint) "
                "should be selected.")

        if eta_p.min() <= 0:
            warnings.warn(
                f"Minimum eta(p) = {eta_p.min()} <= 0, p-Leaders correction "
                "cannot be applied. A smaller value of p (or larger value of "
                "gamint) should be selected.")

        self.eta_p = eta_p

        self.correct_pleaders()

    def plot(self, j1, j2, ax=None, vmin=None, vmax=None, cbar=True,
             figsize=(4.5, 1.5), gamma=.3, nan_idx=None, signal_idx=0,
             cbar_kw=None, cmap='magma'):
        
        if self.eta_p is None and not np.isinf(self.p_exp):
            self.check_regularity([(j1, j2)], None, None)

        super().plot(j1, j2, ax, vmin, vmax, cbar, figsize, gamma,
                     nan_idx, signal_idx, cbar_kw, cmap)
        

@dataclass(kw_only=True)
class WaveletWSE(WaveletDec):
    r"""
    Wavelet Weak Scaling Exponent.

    It represents the :math:`(\theta, \omega)`-leaders of a signal: :math:`\ell^{(\theta, \omega)}(j, k)`.

    .. note:: Should not be instantiated directly but instead created using the :func:`WaveletDec.get_wse` method.

    Attributes
    ----------
    values : dict[int, ndarray]
        ``values[j]`` contains the coefficients at the scale j.
        Arrays are of the shape (nj, n_rep)
    n_sig: int
        Number of underlying signals in the wavelet decomposition. May not
        match the dimensionality of the values arrays in case there are
        multiple repetitions associated to a single signal.
    gamint : float
        Fractional integration used in the computation of the MRQ.
    wt_name : str
        Name of the wavelet used for the decomposition.
    interval_size : int
        Width of the coef interval over which the MRQ was computed.
    origin_mrq : WaveletDec | None
        If MRQ is derived from another mrq, refers to the original MRQ.
    bootstrapped_obj : :class:`.MultiResolutionQuantity` | None
        Storing the bootstrapped version of the MRQ if bootstraping has been
        used.
    theta: float
        Cone spread parameter in computing the WSE.
    """
    theta: float

    def get_formalism(self):
        return 'weak scaling exponent'
    
    def get_wse(self, theta=0.5, gamint=None):

        if (theta == self.theta
                and (gamint is None or gamint == self.gamint)):
            return self

        return super().get_wse(theta, gamint)

    def check_regularity(self, *args):
        """
        Check that the MRQ has enough regularity for analysis

        Returns
        -------
        None
        """
        return None

    def integrate(self, gamint):
        """
        Re-compute the WSE on the fractionally integrated wavelet coefficients.

        Parameters
        ----------
        gamint : float
            Fractional integration coefficient

        Returns
        -------
        integrated : WaveletWSE
        """
        return self.origin_mrq.get_wse(self.theta, gamint)
