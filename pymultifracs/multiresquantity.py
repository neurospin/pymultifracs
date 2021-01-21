"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from dataclasses import dataclass, field
import inspect

import numpy as np


@dataclass
class MultiResolutionQuantityBase:
    formalism: str = field(init=False, default=None)
    nj: dict = field(init=False, default_factory=dict)
    values: dict = field(init=False, default_factory=dict)

    def add_values(self, coeffs, j):

        self.values[j] = coeffs
        self.nj[j] = len(coeffs)
        # self.n_scales += 1

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
    nj : dict
        Contains the number of coefficients at the scale j.
    values : dict
        Values[j] contains the coefficients at the scale j.
    """
    formalism: str

    def __post_init__(self):

        if self.formalism not in ['wavelet coef', 'wavelet leader',
                                  'wavelet p-leader']:
            raise ValueError('formalism needs to be one of : "wavelet coef", '
                             '"wavelet leader", "wavelet p-leader"')
