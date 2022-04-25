"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import dataclass, field
import inspect
import typing

import numpy as np
from sympy import intersecting_product

from .utils import get_filter_length
from .bootstrap import bootstrap, circular_leader_bootstrap, get_empirical_CI,\
    max_scale_bootstrap, get_confidence_interval, get_empirical_variance,\
    get_variance, get_std
from .autorange import compute_Lambda, compute_R, find_max_lambda
from .regression import compute_R2


@dataclass
class MultiResolutionQuantityBase:
    formalism: str = field(init=False, default=None)
    gamint: float = field(init=False, default=None)
    nj: dict = field(init=False, default_factory=dict)
    nrep: int = field(init=False)
    bootstrapped_mrq: typing.Any = field(init=False, default=None)

    def get_nj(self):
        """
        Returns nj as a list
        """
        return list(self.nj.values())

    def get_nj_interv(self, j1, j2):
        """
        Returns nj as a list, for j in [j1,j2]
        """
        # nj = []
        # for j in range(j1, j2+1):
        #     nj.append(self.nj[j])
        # return nj
        return np.array([self.nj[j] for j in range(j1, j2+1)])

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



    def sup_coeffs(self, n_ranges, j_max, j_min, scaling_ranges):

        sup_coeffs = np.ones((j_max - j_min + 1, n_ranges, self.nrep))

        for i, (j1, j2) in enumerate(scaling_ranges):
            for j in range(j1, j2 + 1):
                c_j = np.abs(self.values[j])
                sup_c_j = np.nanmax(c_j, axis=0)
                sup_coeffs[j-j_min, i] = sup_c_j

        return sup_coeffs

    def j2_eff(self):
        return len(self.nj)

    def _get_j_min_max(self):

        j_min = min([sr[0] for sr in self.scaling_ranges])
        j_max = max([sr[1] for sr in self.scaling_ranges])

        return j_min, j_max

    def _compute_R2(self, moment, slope, intercept, weights):
        return compute_R2(moment, slope, intercept, weights, *self._get_j_min_max())

    def _compute_R(self, moment, slope, intercept):
        return compute_R(moment, slope, intercept, *self._get_j_min_max())

    def compute_Lambda(self):

        R = self.compute_R()
        R_b = self.bootstrapped_mrq.compute_R()

        return compute_Lambda(R, R_b)

    def find_best_range(self):
        return find_max_lambda(self.compute_Lambda())

    def _check_enough_rep_bootstrap(self):

        if self.nrep < 2:
            raise ValueError(
                f'nrep={self.nrep} too small to build confidence intervals'
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
                "Bootstrapped mrq needs to be computed prior to estimating empirical estimators")
        
        self.bootstrapped_mrq._check_enough_rep_bootstrap()

    def __getattr__(self, name):

        if name[:3] == 'CI_':

            bootstrapped_mrq = self._get_bootstrapped_mrq()

            return get_confidence_interval(bootstrapped_mrq, name[3:])

        elif name[:4] == 'CIE_':

            self._check_bootstrap_mrq()

            return get_empirical_CI(self.bootstrapped_mrq, self, name[4:])

        elif name[:3] == 'VE_':

            self._check_bootstrap_mrq()
        
            return get_empirical_variance(self.bootstrapped_mrq, self, name[3:])

        elif name[:3] == 'SE_':

            self._check_bootstrap_mrq()

            return np.sqrt(
                get_empirical_variance(self.bootstrapped_mrq, self, name[3:](self)))

        elif name[:2] == 'V_':

            bootstrapped_mrq = self._get_bootstrapped_mrq()

            return get_variance(bootstrapped_mrq, name[2:])

        elif name[:4] == 'STD_':

            bootstrapped_mrq = self._get_bootstrapped_mrq()

            return get_std(bootstrapped_mrq, name[4:])

        return None


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
        Arrays are of the shape (nrep,)
    values : dict(ndarray)
        `values[j]` contains the coefficients at the scale j.
        Arrays are of the shape (nj, nrep)
    nrep : int
        Number of realisations
    """
    formalism: str
    gamint: float
    values: dict = field(default_factory=dict)
    nj: dict = field(default_factory=dict)
    bootstrapped_mrq: MultiResolutionQuantityBase = field(init=False,
                                                          default=None)

    def __post_init__(self):

        if self.formalism not in ['wavelet coef', 'wavelet leader',
                                  'wavelet p-leader']:
            raise ValueError('formalism needs to be one of : "wavelet coef", '
                             '"wavelet leader", "wavelet p-leader"')

    def bootstrap(self, R, wt_name):

        if self.formalism == 'wavelet coef':

            # print("Using coef bootstrapping technique")
            self.bootstrapped_mrq = bootstrap(self, R, wt_name)
            return self.bootstrapped_mrq

        elif 'leader' in self.formalism:

            # print("Using leader bootstrapping technique")

            block_length = get_filter_length(wt_name)
            max_scale = max_scale_bootstrap(self, block_length)

            if max_scale < self.j2_eff():
                raise ValueError(f'Maximum bootstrapping scale {max_scale} is '
                                 f'inferior to the j2={self.j2_eff()} chosen '
                                 'when computing wavelet leaders.')

            self.bootstrapped_mrq = circular_leader_bootstrap(
                self, max_scale, block_length, R)
            return self.bootstrapped_mrq

    def add_values(self, coeffs, j):

        self.values[j] = coeffs
        self.nj[j] = (~np.isnan(coeffs)).sum(axis=0)

    def __getattr__(self, name):
        if name == 'nrep':
            if len(self.values) > 0:
                return self.values[[*self.values][0]].shape[1]

        return self.__getattribute__(name)
