"""
JAX-accelerate simulation routines
"""

import jax
import jax.numpy as jnp
from pymultifracs.simul.pzutils import generate_z0, generate_temporal_cov_fbm


_generate_z0 = jax.jit(generate_z0, static_argnames=['N', 'R'])
_generate_temporal_cov_fbm = jax.jit(generate_temporal_cov_fbm, static_argnames=['N'])


@jax.jit(static_argnames=['N', 'R'])
def _gaussian_cme(cov, N, R, z):
    """
    JAX-accelerated version of :fun:`gaussian_cme`. Automatically used when jax
    is detected.
    """
    
    cov = jnp.concatenate((cov, jnp.flip(cov[1:-1])), axis=0)
    
    L = jnp.fft.fft(cov)
    
    x = jnp.fft.fft(
        z * jnp.sqrt(L / (2 * N - 2)),
        axis=0).real
    
    return x[:N], any(L.real < 0)  #x, warning flag if negative fft coef


def _skewness_conv(N, K0, dt, tau, alpha, beta, e):
    """
    e: noise
    """
    
    Kbar = jnp.zeros(2 * N)
    Kbar.at[1:N+1].set(K0 / tau ** alpha / dt ** beta)
    
    return jnp.fft.ifft(
        jnp.fft.fft(
            Kbar * jnp.fft.fft(e, axis=0)
        ), axis=0
    ).real[N:]