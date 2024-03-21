from dataclasses import dataclass, field, InitVar
import inspect

import numpy as np

from .multiresquantity import WaveletDec


@dataclass
class ScalingFunction:
    mrq: InitVar[WaveletDec]
    idx_reject: InitVar[dict[int, np.ndarray]] = field(default=None)
    scaling_ranges: list[tuple[int]]
    weighted: str = None
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
