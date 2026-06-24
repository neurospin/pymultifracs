import jax.numpy as jnp


def compute_etap(wt_coefs, j1, j2, p, validity):

    Sj = jnp.zeros((j2-j1+1, wt_coefs[j1].shape[1]))

    Sj = jnp.stack([
        jnp.log2(jnp.mean(wt_coefs[scale] ** p, axis=0, where=validity[scale]))
        # jnp.log2(jnp.nanmean(wt_coefs[scale] ** p, axis=0))
        for scale in range(j1, j2+1)
    ], axis=0)

    return jnp.polyfit(jnp.arange(j1, j2+1).astype(float), Sj, 1)[0]
