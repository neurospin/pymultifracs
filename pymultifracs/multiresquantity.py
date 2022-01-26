"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import dataclass, field
import inspect

import numpy as np

from .utils import get_filter_length
from .bootstrap import bootstrap, circular_leader_bootstrap, get_empirical_CI,\
    max_scale_bootstrap, get_confidence_interval


@dataclass
class MultiResolutionQuantityBase:
    formalism: str = field(init=False, default=None)
    gamint: float = field(init=False, default=None)
    nj: dict = field(init=False, default_factory=dict)
    nrep: int = field(init=False)

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

    def bootstrap(self, R, wt_name):

        if self.formalism == 'wavelet coef':

            print("Using coef bootstrapping technique")
            return bootstrap(self, R, wt_name)

        elif 'leader' in self.formalism:

            print("Using leader bootstrapping technique")

            block_length = get_filter_length(wt_name)
            max_scale = max_scale_bootstrap(self, block_length)

            if max_scale < self.j2_eff():
                raise ValueError(f'Maximum bootstrapping scale {max_scale} is '
                                 f'inferior to the j2={self.j2_eff()} chosen '
                                 'when computing wavelet leaders.')

            return circular_leader_bootstrap(self, max_scale, block_length, R)

    def j2_eff(self):
        return len(self.nj)

    def __getattr__(self, name):

        if name[:3] == 'CI_':
            if self.nrep < 2:
                raise ValueError(
                    f'nrep={self.nrep} too small to build confidence intervals'
                    )

            return get_confidence_interval(self, name[3:])

        if name[:4] == 'CIE_':

            if self.nrep < 2:
                raise ValueError(
                    f'nrep={self.nrep} too small to build confidence intervals'
                    )

            def wrapper(ref_mrq):

                return get_empirical_CI(self, ref_mrq, name[4:])

            return wrapper

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

    def __post_init__(self):

        if self.formalism not in ['wavelet coef', 'wavelet leader',
                                  'wavelet p-leader']:
            raise ValueError('formalism needs to be one of : "wavelet coef", '
                             '"wavelet leader", "wavelet p-leader"')

    def add_values(self, coeffs, j):

        self.values[j] = coeffs
        self.nj[j] = (~np.isnan(coeffs)).sum(axis=0)

    def __getattr__(self, name):
        if name == 'nrep':
            if len(self.values) > 0:
                return self.values[[*self.values][0]].shape[1]

        return self.__getattribute__(name)
