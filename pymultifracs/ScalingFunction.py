from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScalingFunction:
    weights: np.ndarray = field(init=False, default=None)
    
