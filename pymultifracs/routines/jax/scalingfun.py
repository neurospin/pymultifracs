import jax.numpy as jnp


def _compute_Cmj(pleader_p, j, p, ZPJCorr, validity, n_cumul, bias=True):

    T_X_j = jnp.log(pleader_p ** (1/p) * ZPJCorr)

    Mu = jnp.stack([
        jnp.mean(T_X_j ** m, where=validity, axis=0)
        for m in range(1, n_cumul+1)
    ], axis=0)

    Cmj = jnp.zeros_like(Mu)

    for ind_m, m in enumerate(range(1, n_cumul+1)):

        aux = 0

        for ind_n, n in enumerate(range(1, m)):
            aux = aux + _binom(m-1, n-1) * Cmj[ind_n] * Mu[ind_m-ind_n-1]

        Cmj = Cmj.at[ind_m].set(Mu[ind_m] - aux)
    
    if not bias:

        N_useful = validity.sum(axis=0)
        correction_factor = jnp.ones_like(N_useful, dtype=float)

        for m in range(2, n_cumul+1):

            ind_m = m-1
            correction_factor = correction_factor * N_useful / (N_useful - (m-1)) #* 10
            Cmj = Cmj.at[ind_m].set(Cmj[ind_m] * correction_factor)

    return Cmj


def _compute_Cm_jax(pleader_p, j1, j2, n_cumul, p, ZPJCorr, validity, bias=True):

    Cm = jnp.stack(
        [
            _compute_Cmj(pleader_p[j], j, p, ZPJCorr[:, j-1], validity[j],
                         n_cumul, bias)
            for ind_j, j in enumerate(range(j1, j2+1))
        ],
        axis=0
    )

    # shape j, m, channel
    return Cm
