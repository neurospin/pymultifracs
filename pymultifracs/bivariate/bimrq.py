from dataclasses import dataclass, field

import numpy as np


@dataclass
class BiMRQBase:
    formalism: str = field(init=False)
    nj: np.ndarray = field(init=False)
    j: np.ndarray = field(init=False)
    j1: int  # = field(init=False)
    j2: int  # = field(init=False)
    wtype: bool  # = field(init=False)
