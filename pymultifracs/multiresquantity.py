"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import dataclass, field
import inspect
from typing import Any

import numpy as np

from .utils import get_filter_length, max_scale_bootstrap, _correct_pleaders,\
    mask_reject
from .autorange import compute_Lambda, compute_R, find_max_lambda
from .regression import compute_R2


@dataclass
class MultiResolutionQuantityBase:
    formalism: str = field(init=False, default=None)
    gamint: float = field(init=False, default=None)
    wt_name: str = field(init=False, default=None)
    nj: dict = field(init=False, default_factory=dict)
    n_sig: int = field(init=False, default=None)
    bootstrapped_mrq: Any = field(init=False, default=None)
    origin_mrq: Any = field(init=False, default=None)

    def get_nj(self):
        """
        Returns nj as a list
        """
        return list(self.nj.values())

    def get_nj_interv(self, j1, j2):
        """
        Returns nj as an array, for j in [j1,j2]
        """
        return np.array([self.nj[j] for j in range(j1, j2+1)])

    def update_nj(self):
        self.nj = {
            scale: (~np.isnan(self.values[scale])).sum(axis=0)
            for scale in self.values
        }

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

    def sup_coeffs(self, n_ranges, j_max, j_min, scaling_ranges, idx_reject):

        sup_coeffs = np.ones((j_max - j_min + 1, n_ranges, self.n_rep))

        for i, (j1, j2) in enumerate(scaling_ranges):
            for j in range(j1, j2 + 1):

                c_j = np.abs(self.values[j])

                c_j = mask_reject(c_j, idx_reject, j, 1)
                
                sup_c_j = np.nanmax(c_j, axis=0)
                sup_coeffs[j-j_min, i] = sup_c_j

        return sup_coeffs

    def j2_eff(self):
        return max(list(self.nj))

    def _get_j_min_max(self):

        j_min = min([sr[0] for sr in self.scaling_ranges])
        j_max = max([sr[1] for sr in self.scaling_ranges])

        return j_min, j_max

    def _compute_R2(self, moment, slope, intercept, weights):
        return compute_R2(moment, slope, intercept, weights,
                          [self._get_j_min_max()], self.j)

    def _compute_R(self, moment, slope, intercept):
        return compute_R(moment, slope, intercept,
                         [self._get_j_min_max()], self.j)

    def compute_Lambda(self):

        R = self.compute_R()
        R_b = self.bootstrapped_mrq.compute_R()

        print(R.shape, R_b.shape)

        return compute_Lambda(R, R_b)

    def find_best_range(self):
        return find_max_lambda(self.compute_Lambda())

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

        print(bootstrapped_mrq.n_rep, bootstrapped_mrq.S2.shape)

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


@dataclass
class MultiResolutionQuantity(MultiResolutionQuantityBase):
    """
    Handles multi-resolution quantities in multifractal analysis.

    It can be used to represent wavelet coefficients :math:`d_X(j, k)`
    and wavelet leaders :math:`L_X(j, k)`.

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
    n_scales : int
        Size of the scale range covered.
    nj : dict(ndarray)
        Contains the number of coefficients at the scale j.
        Arrays are of the shape (n_rep,)
    values : dict(ndarray)
        `values[j]` contains the coefficients at the scale j.
        Arrays are of the shape (nj, n_rep)
    n_rep : int
        Number of realisations
    """
    formalism: str
    gamint: float
    wt_name: str
    n_sig: int = None
    p_exp: float = None
    interval_size: int = 1
    values: dict = field(default_factory=dict)
    nj: dict = field(default_factory=dict)
    origin_mrq: MultiResolutionQuantityBase = None
    eta_p: np.ndarray = field(init=False, default=None)
    ZPJCorr: np.ndarray = field(init=False, default=None)
    bootstrapped_mrq: MultiResolutionQuantityBase = field(init=False,
                                                          default=None)

    def __post_init__(self):

        if self.formalism not in ['wavelet coef', 'wavelet leader',
                                  'wavelet p-leader', 'weak scaling exponent']:
            raise ValueError('formalism needs to be one of : "wavelet coef", '
                             '"wavelet leader", "wavelet p-leader"')

    def bootstrap(self, R, min_scale=1):

        from .bootstrap import circular_leader_bootstrap

        block_length = get_filter_length(self.wt_name)
        max_scale = max_scale_bootstrap(self)

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
        self.nj[j] = (~np.isnan(coeffs)).sum(axis=0)

    def correct_pleaders(self, min_scale, max_scale):

        self.ZPJCorr = _correct_pleaders(
            self, self.p_exp, min_scale, max_scale)

        return self.ZPJCorr

    def __getattribute__(self, name: str) -> Any:

        if name == 'filt_len':
            return get_filter_length(self.wt_name)

        if name == 'n_sig' and super().__getattribute__('n_sig') is None:
            return 1

        return super().__getattribute__(name)

    def __getattr__(self, name):

        if name == 'n_rep':
            if len(self.values) > 0:
                return self.values[[*self.values][0]].shape[1]

        return super().__getattr__(name)
