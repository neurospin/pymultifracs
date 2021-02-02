from dataclasses import dataclass, field

import numpy as np


@dataclass
class BiMRQBase:
    wtype: bool = field(init=False)
    formalism: str = field(init=False)
    nj: np.ndarray = field(init=False)
    j: np.ndarray = field(init=False)
