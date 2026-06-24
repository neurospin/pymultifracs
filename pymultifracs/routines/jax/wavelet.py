"""
Authors: Merlin Dumeur <merlin@dumeur.net>
"""

import jax.numpy as jnp


# def _integrate(wt_coefs_values, gamint, max_level):
# 
#     wt_out = {}
# 
#     for scale in range(1, max_level + 1):
# 
#         wt_out[scale] = wt_coefs_values[scale] * 2 ** (gamint * scale)
# 
#     return wt_out


def _compute_leaders_jax(wt_coefs_values, p_exp, size, max_level, leader_flag):

    pleader_p = {}

    for scale in range(1, max_level + 1):

        coefs = jnp.power(jnp.abs(wt_coefs_values[scale]), p_exp)

        scale_contribution = jnp.zeros((size, *coefs.shape))  # - 1

        if size > 1:
            idx_size = np.s_[(size-1)//2:-((size-1)//2)]
        else:
            idx_size = np.s_[:]

        scale_contribution = scale_contribution.at[:, idx_size].set(
            jnp.stack([
                coefs[size-i:-(i-1) or None] for i in range(1, size+1)
            ], axis=0))

        if scale == 1:

            leaders = np.sum(scale_contribution, axis=0)
            pleader_p[scale] = leaders

            continue

        max_index = pleader_p[scale-1].shape[0] // 2

        lower_contribution = jnp.zeros((2, *coefs.shape))  # - 1

        lower_contribution = lower_contribution.at[0, :max_index].set(
            pleader_p[scale-1][::2][:max_index]
        )
        lower_contribution = lower_contribution.at[1, :max_index].set(
            pleader_p[scale-1][1::2][:max_index]
        )

        if leader_flag:

            pleader_p[scale] = jnp.max(np.r_[
                scale_contribution,
                .5 * lower_contribution
            ], axis=0)

        else:

            pleader_p[scale] = jnp.sum(jnp.r_[
                scale_contribution,
                .5 * lower_contribution
            ], axis=0)  # * ZPJCorr[:, scale-1]

    return pleader_p


def _correct_pleaders_jax(eta_p, p_exp, max_level):

    j_array = jnp.arange(1, max_level + 1)
    JJ = j_array
    J1LF = 1
    JJ0 = JJ - J1LF + 1

    # eta_p shape (n_rep,)
    # JJ0 shape (n_level,)

    JJ0 = JJ0[None, :]
    # eta_p = wt_leaders.eta_p
    eta_p = eta_p[:, None]

    zqhqcorr = jnp.log2((1 - jnp.power(2., -JJ0 * eta_p))
                        / (1 - jnp.power(2., -eta_p)))
    ZPJCorr = jnp.power(2, (-1.0 / p_exp) * zqhqcorr)

    # ZPJCorr shape (n_ranges, n_rep, n_level)
    # ZPJCorr = ZPJCorr.at[eta_p <= 0].set(1)
    ZPJCorr = jnp.where(eta_p < 0, 1, ZPJCorr)

    return ZPJCorr


def _integrate_wavelet(wt_coefs, gamint):

    return {
        scale: wt_coefs[scale] * 2 ** (scale * gamint)
        for scale in wt_coefs
    }