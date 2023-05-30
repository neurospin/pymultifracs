from dataclasses import dataclass, field, InitVar

import numpy as np

from ..multiresquantity import MultiResolutionQuantityBase, \
    MultiResolutionQuantity


@dataclass
class BiMultifractalSpectrum(MultiResolutionQuantityBase):
    mrq1: InitVar[MultiResolutionQuantity]
    mrq2: InitVar[MultiResolutionQuantity]
    j1: int
    j2: int
    weighted: bool
    q1: np.ndarray
    q2: np.ndarray
    j: np.ndarray = field(init=False)
    Dq: np.array = field(init=False)
    hq: np.array = field(init=False)
    U: np.array = field(init=False)
    V: np.array = field(init=False)

    def __post_init__(self, mrq1, mrq2):

        self.nrep = 1
        self.n_sig = 1

        assert mrq1.formalism == mrq2.formalism
        self.formalism = mrq1.formalism

        assert mrq1.nj == mrq2.nj
        self.nj = mrq1.nj
        self.j = np.array(list(mrq1.values))

        self._compute(mrq1, mrq2)

    def _compute(self, mrq1, mrq2):
        pass
