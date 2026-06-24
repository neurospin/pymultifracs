import jax.numpy as jnp
from jax.scipy import special


def _binom(N, k):
    return jnp.exp(
        special.gammaln(N + 1)
        - special.gammaln(k + 1)
        - special.gammaln(N - k + 1)
    )


def get_validity_coef(values_dict):

    return {
        scale: ~np.isnan(val)
        for scale, val in values_dict.items()
    }


def get_validity_leader(values_dict):

    return {
        scale: ~(np.isnan(val) | (val < 0))
        for scale, val in values_dict.items()
    }
