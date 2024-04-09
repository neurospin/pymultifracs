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

    def from_dict(self, d):
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
    
    def scale2freq(self, scale, sfreq):
        return pywt.scale2frequency(self.wt_name, scale) * sfreq
        
    def freq2scale(self, freq, sfreq):
        return pywt.frequency2scale(self.wt_name, freq / sfreq)
    
    def max_scale_bootstrap(self, idx_reject=None):
        return max_scale_bootstrap(self, idx_reject)


@dataclass(kw_only=True)
class WaveletDec(MultiResolutionQuantityBase):
    r"""
    Handles multi-resolution quantities in multifractal analysis.

    It can be used to represent wavelet coefficients :math:`d_X(j, k)`
    and wavelet (p-)leaders :math:`L_X(j, k)`.

    Parameters
    ----------
    formalism : str
        Indicates the formalism used to obtain the multi resolution quantity.
        Can be any of 'wavelet coef', 'wavelet leader',
        or 'wavelet p-leaders'.

    Attributes
    ----------
    formalism : str
        Formalism used. Can be any of 'wavelet coef', 'wavelet leader',
        or 'wavelet p-leaders'.
    gamint : float
        Fractional integration used in the computation of the MRQ.
    wt_name : str
        Name of the wavelet used for the MRQ.
    p_exp : float | None
        Optional, for wavelet p-leaders indicates the p-exponent, takes value
        np.inf for wavelet leaders.
    interval_size : int
        Width of the coef interval over which the MRQ was computed.
    values : dict(ndarray)
        `values[j]` contains the coefficients at the scale j.
        Arrays are of the shape (nj, n_rep)
    nj : dict(ndarray)
        Contains the number of coefficients at the scale j.
        Arrays are of the shape (n_rep,).
    origin_mrq : MultiResolutionQuantity | None
        If MRQ is derived from another mrq, refers to the original MRQ.
    eta_p : float | None
        Only for p-leaders, wavelet scaling function :math:`\eta(p)`.
        By default only computed during mf_analysis.
    ZPJCorr : ndarray | None
        Only for p-leaders, correction factor for the finite size effects,
        dependent on `eta_p`.
    bootstrapped_obj : :class:`.MultiResolutionQuantity` | None
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

        out = self.values[j][:, None, :]
        
        # Bootstrapped mrq needs to realign into signal and repetitions
        if reshape: # and self.n_rep != self.n_sig
            out = out.reshape(self.values[j].shape[0], 1, self.n_sig, -1)

        if idx_reject is None:
            return out

        return mask_reject(
            out, idx_reject, j, self.interval_size)

    def plot(self, j1, j2, **kwargs):
        viz.plot_coef(self, j1, j2, **kwargs)

    def get_formalism(self):
        return 'wavelet coef'
    
    def integrate(self, gamint):
        return wavelet.integrate_wavelet(self, gamint)
    
    def get_leaders(self, p_exp, interval_size=3, gamint=0):

        if self.origin_mrq is not None:
            return self.origin_mrq.get_leaders(p_exp, interval_size, gamint)
        
        if gamint != 0 and gamint != self.gamint:
            
            int = self.integrate(gamint)

            return wavelet.compute_leaders(int, p_exp, interval_size)
        
        return wavelet.compute_leaders(
            self, p_exp, interval_size)
    
    def get_wse(self, theta=.5, gamint=0):

        if self.origin_mrq is not None:
            return self.origin_mrq.get_wse(theta, gamint)
        
        return wavelet.compute_wse(self, theta, gamint)
    
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

    def _check_regularity(self, scaling_ranges, weighted=None,
                          idx_reject=None):
        
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
                return self.values[[*self.values][0]].shape[1]

        # if name == 'n_sig' and super().__getattribute__('n_sig') is None:
        #     return 1

        return super().__getattribute__(name)

    # def __getattr__(self, name):
    #     return super().__getattr__(name)


@dataclass(kw_only=True)
class WaveletLeader(WaveletDec):
    p_exp: float
    interval_size: int = 1
    eta_p: np.ndarray = field(init=False, repr=False)
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
        if self.p_exp == np.inf:
            return 'wavelet leader'
        return 'wavelet p-leader'
    
    def integrate(self, gamint):
        return self.get_leaders(self.p_exp, self.interval_size, gamint)

    def get_values(self, j, idx_reject=None, reshape=False):

        if self.p_exp == np.inf or self.eta_p is None:
            return super().get_values(j, idx_reject, reshape)

        if self.ZPJCorr is None:
            self.correct_pleaders()

        ZPJCorr = self.ZPJCorr[None, :, :, j - min(self.values)]

        if reshape:
            ZPJCorr = ZPJCorr[..., None]

        return ZPJCorr * super().get_values(j, idx_reject, reshape)

    def get_leaders(self, p_exp, interval_size=3, gamint=0):

        if (p_exp == self.p_exp
            and interval_size == self.interval_size
            and gamint == self.gamint):

            return self

        return self.origin_mrq.get_leaders(p_exp, interval_size, gamint)

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

    def _check_regularity(self, scaling_ranges, weighted=None,
                          idx_reject=None):

        if self.p_exp == np.inf:
            return super()._check_regularity(
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

@dataclass(kw_only=True)
class Wtwse(WaveletDec):
    theta: float

    def get_formalism(self):
        return 'weak scaling exponent'

    def _check_regularity(self, *args):
        return None
