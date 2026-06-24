import jax.numpy as jnp


def _compute_weights(validity, j1, j2):

    return jnp.stack([
        jnp.sqrt(validity[j][:, 0].sum()) for j in range(j1, j2+1)
    ])


def _compute_cm_jax(Cm, j_array, n_cumul, weights=None):

    fits = [
        jnp.polyfit(j_array.astype(float), Cm[:, m], 1, w=weights, full=True)
        for m in range(n_cumul)
    ]

    cm = jnp.stack([f[0][0] for f in fits], axis=0)
    cm0 = jnp.stack([f[0][1] for f in fits], axis=0)
    resids = jnp.stack([f[1] for f in fits], axis=0)

    return cm, cm0, resids
