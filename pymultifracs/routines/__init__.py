import os

from ..backend import JAX_AVAILABLE

if JAX_AVAILABLE:
    from .jax.utils import _binom
    from .jax.wavelet import _compute_leaders_jax as _compute_leaders
    from .jax.wavelet import  _correct_pleaders_jax as _correct_pleaders
    from .jax.scalingfun import _compute_Cmj, _compute_Cm_jax
    from .jax.estimation import compute_etap
    from .jax.mfa import _compute_weights, _compute_cm_jax
    from .jax.simul import _gaussian_cme, _skewness_conv
else:
    pass
