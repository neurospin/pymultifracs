import os

JAX_AVAILABLE = False
PYMFA_USE_JAX = bool(os.getenv('PYMFA_USE_JAX', 1))

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    jax = None

if JAX_AVAILABLE and PYMFA_USE_JAX:
    import jax.numpy as jnp
    from jax import jit, vmap
    from jax.lax import cond as lax_cond

    print(f'JAX:{jax.__version__} found, using JAX routines')

    JAX_AVAILABLE = True

    try:
        import xarray_jax
    except ImportError:
        print('xarray_jax not found, some features may be broken')

else:

    if JAX_AVAILABLE:
        print('PYMFA_USE_JAX set to 0, using numpy routines')
    else:
        print('JAX not found, using numpy routines')

    import numpy as jnp

    def jit(f, *a, **k): return f
    def vmap(f, *a, **k): return f

    def lax_cond(cond, true_fun, false_fun, *operands):

        if cond:
            return true_fun(*operands)

        return false_fun(*operands)
