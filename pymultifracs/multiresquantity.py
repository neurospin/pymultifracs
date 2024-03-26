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
    mask_reject
from . import viz, wavelet, estimation

@dataclass
class MultiResolutionQuantityBase:
    n_sig: int
    bootstrapped_mrq: Any | None = None
    origin_mrq: Any | None = None

    # def get_nj(self):
    #     """
    #     Returns nj as a list
    #     """
    #     return list(self.nj.values())

    # def get_nj_interv(self, j1, j2):
    #     """
    #     Returns nj as an array, for j in [j1,j2]
    #     """
    #     return np.array([self.nj[j] for j in range(j1, j2+1)])

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

    def _get_j_min_max(self):

        j_min = min([sr[0] for sr in self.scaling_ranges])
        j_max = max([sr[1] for sr in self.scaling_ranges])

        return j_min, j_max
    
    def _check_enough_rep_bootstrap(self):

        if (ratio := self.n_rep // self.n_sig) < 2:
            raise ValueError(
                f'n_rep = {ratio} per original signal too small to build '
                'confidence intervals'
                )

    def _get_bootstrapped_mrq(self):

        if self.bootstrapped_mrq is None:
            bootstrapped_mrq = self
        else:
            bootstrapped_mrq = self.bootstrapped_mrq

        bootstrapped_mrq._check_enough_rep_bootstrap()

        return bootstrapped_mrq

    def _check_bootstrap_mrq(self):

        if self.bootstrapped_mrq is None:
            raise ValueError(
                "Bootstrapped mrq needs to be computed prior to estimating "
                "empirical estimators")

        self.bootstrapped_mrq._check_enough_rep_bootstrap()

    def __getattr__(self, name):

        if name[:3] == 'CI_':
            from .bootstrap import get_confidence_interval

            bootstrapped_mrq = self._get_bootstrapped_mrq()

            return get_confidence_interval(bootstrapped_mrq, name[3:])

        elif name[:4] == 'CIE_':
            from .bootstrap import get_empirical_CI

            self._check_bootstrap_mrq()

            return get_empirical_CI(self.bootstrapped_mrq, self, name[4:])

        elif name[:3] == 'VE_':
            from .bootstrap import get_empirical_variance

            self._check_bootstrap_mrq()

            return get_empirical_variance(self.bootstrapped_mrq, self,
                                          name[3:])

        elif name[:3] == 'SE_':

            from .bootstrap import get_empirical_variance

            self._check_bootstrap_mrq()

            return np.sqrt(
                get_empirical_variance(self.bootstrapped_mrq, self,
                                       name[3:](self)))

        elif name[:2] == 'V_':

            from .bootstrap import get_variance

            bootstrapped_mrq = self._get_bootstrapped_mrq()

            return get_variance(bootstrapped_mrq, name[2:])

        elif name[:4] == 'STD_':

            from .bootstrap import get_std

            bootstrapped_mrq = self._get_bootstrapped_mrq()

            return get_std(bootstrapped_mrq, name[4:])

        return self.__getattribute__(name)
    
    def scale2freq(self, scale, sfreq):
        return pywt.scale2frequency(self.wt_name, scale) * sfreq
        
    def freq2scale(self, freq, sfreq):
        return pywt.frequency2scale(self.wt_name, freq / sfreq)


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
    bootstrapped_mrq : :class:`.MultiResolutionQuantity` | None
        Storing the bootstrapped version of the MRQ if bootstraping has been
        used.
    """
    wt_name: str
    gamint: float = 0
    values: dict = field(default_factory=dict)
    origin_mrq: MultiResolutionQuantityBase | None = None
    interval_size: int = field(init=False, default=1)

    def bootstrap(self, R, min_scale=1, idx_reject=None):

        from .bootstrap import circular_leader_bootstrap

        block_length = get_filter_length(self.wt_name)
        max_scale = max_scale_bootstrap(self, idx_reject)

        self.bootstrapped_mrq = circular_leader_bootstrap(
            self, min_scale, max_scale, block_length, R)

        # j = np.array([*self.values])
        #
        # if min_scale > j.min():
        #     self.values = {scale: value
        #                    for scale, value in self.values.items()
        #                    if scale >= min_scale}
        #     self.nj = {scale: nj for scale, nj in self.nj.items()
        #                if scale >= min_scale}

        return self.bootstrapped_mrq

    @classmethod
    def bootstrap_multiple(cls, R, min_scale, mrq_list):

        from .bootstrap import circular_leader_bootstrap

        block_length = max([
            get_filter_length(mrq.wt_name) for mrq in mrq_list
        ])

        max_scale = min([
            max_scale_bootstrap(mrq) for mrq in mrq_list
        ])

        j2_eff = np.array([mrq.j2_eff() for mrq in mrq_list])
        wrong_idx = max_scale < j2_eff

        if wrong_idx.any():
            raise ValueError(f'Maximum bootstrapping scale {max_scale} is '
                             f'inferior to the j2 chosen when computing '
                             f'wavelet leaders for indices {wrong_idx}.')

        return circular_leader_bootstrap(mrq_list, min_scale, max_scale,
                                         block_length, R)

    def add_values(self, coeffs, j):
        self.values[j] = coeffs

    def get_values(self, j, idx_reject=None, reshape=False):

        out = self.values[j][:, None, :]
        
        # Bootstrapped mrq needs to realign into signal and repetitions
        if self.n_rep != self.n_sig and reshape:
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

            if p_exp is None:
                return self

            return wavelet.compute_leaders(int, p_exp, interval_size)
        
        return wavelet.compute_leaders(
            self, p_exp, interval_size)
    
    def get_wse(self, theta=.5, gamint=0):

        if self.origin_mrq is not None:
            return self.origin_mrq.get_wse(theta, gamint)
        
        return wavelet.compute_wse(self, theta, gamint)

    def _check_regularity(self, scaling_ranges, weighted, idx_reject,
                          gamint=None):
        
        hmin, _ = estimation.estimate_hmin(
            self, scaling_ranges, weighted, idx_reject)
        
        if isinstance(gamint, str) and gamint == 'auto':
            if hmin // .5 > 0:
                gamint = 0
            else:
                gamint = -.5 * (hmin.min() // .5)

                if gamint + hmin < 0.25:
                    gamint += .5

            return gamint

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
    eta_p: np.ndarray = field(init=False)
    ZPJCorr: np.ndarray = field(init=False, default=None)

    def get_formalism(self):
        if self.p_exp == np.inf:
            return 'wavelet leader'
        return 'wavelet p-leader'
    
    def get_values(self, j, idx_reject=None, reshape=False):

        if self.p_exp == np.inf or self.eta_p is None:
            return super().get_values(j, idx_reject)

        if self.ZPJCorr is None:
            self.correct_pleaders(min(self.values), max(self.values))

        ind_j = j - min(self.values)

        # if reshape and self.n_rep != self.n_sig:
        #     ZPJCorr = ZPJCorr[..., None]

        return self.ZPJCorr[None, :, :, ind_j] * super().get_values(j, idx_reject)

    def get_leaders(self, p_exp, interval_size=3, gamint=0):

        if (p_exp == self.p_exp
            and interval_size == self.interval_size
            and gamint == self.gamint):

            return self

        return self.origin_mrq.get_leaders(p_exp, interval_size, gamint)

    def correct_pleaders(self, min_scale, max_scale):

        # No correction if infinite p
        if self.p_exp == np.inf:
            return

        self.ZPJCorr = _correct_pleaders(
            self, self.p_exp, min_scale, max_scale)
        
        return self.ZPJCorr

    def _check_regularity(self, scaling_ranges, weighted, idx_reject,
                          gamint=None):

        if self.p_exp == np.inf:
            return super()._check_regularity(
                scaling_ranges, weighted, idx_reject, gamint)

        eta_p = estimation._estimate_eta_p(
            self.origin_mrq, self.p_exp, scaling_ranges, weighted, idx_reject)

        if isinstance(gamint, str) and gamint == 'auto':
            # gamint = -.5 * (eta_p // .5)
            # gamint[eta_p // .5 > 0] = 0
            # gamint[(gamint + eta_p) < 0.25] += .5
            if eta_p // .5 > 0:
                gamint = 0
            else:
                gamint = -.5 * (eta_p.min() // .5)

                if gamint + eta_p < 0.25:
                    gamint += .5

            return gamint

        if eta_p.max() <= 0:
            # raise ValueError(
            warnings.warn(
                f"Maximum eta(p) = {eta_p.max()} <= 0, no signal can be "
                "analyzed. A smaller value of p (or larger value of gamint) "
                "should be selected.")

        if eta_p.min() <= 0:
            warnings.warn(
                f"Minimum eta(p) = {eta_p.min()} <= 0, p-Leaders correction "
                "cannot be applied. A smaller value of p (or larger value of "
                "gamint) should be selected.")

        self.eta_p = eta_p

        self.correct_pleaders(min([*self.values]), max([*self.values]))

@dataclass(kw_only=True)
class Wtwse(WaveletDec):
    theta: float

    def get_formalism(self):
        return 'weak scaling exponent'

    def _check_regularity(self, *args):
        return None
