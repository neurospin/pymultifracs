"""
Authors: Merlin Dumeur <merlin@dumeur.net>
"""

import numpy as np

from .fbm import fbm
from .mrw import mrw


def create_mask3(length, count, size, align_scale):
    """
    Create a mask of evenly spaced spikes aligned to a specific scale.
    """

    size = int(size)
    mask = np.zeros(length, dtype=bool)

    if count == 0:
        return mask

    coef_edge_pad = 4

    n_coefs_scale = (length // 2 ** align_scale) - 2 * coef_edge_pad

    assert count <= n_coefs_scale

    spacing = n_coefs_scale // (count)
    offset = (n_coefs_scale - (count - 1) * spacing - 1) // 2

    # extra_offset = 0
    # if (n_coefs_scale - (count - 1) * spacing - 1) % 2 == .5:
    #     extra_offset += 1

    if spacing < 3:
        points = np.arange(0, n_coefs_scale, 3)
        cursor = 1
        while points.shape[0] < count:
            points = np.unique(np.r_[points, cursor])
            cursor += spacing

            if cursor == n_coefs_scale:
                cursor = 0
                spacing -= 1
    else:
        points = np.arange(0+offset, n_coefs_scale-offset, spacing)

    assert len(points) == count

    for i in points:
        midpoint = int((i + coef_edge_pad + .5) * 2 ** align_scale)
        mask[midpoint - size // 2:midpoint + size // 2] = 1

    return mask


def generate_simuls_bb(N, lambd=None):
    """
    Generate simulations for a noisy signal with broadband noise.
    """

    if lambd is None:
        X = fbm(shape=N, H=.8)
    else:
        X = mrw(N, .8, lambd, N)
    X_noise = fbm(shape=X.shape[0], H=.5)

    return np.diff(X), np.diff(X_noise)


def gen_noisy(signal, noise, coverage, SNR, align_scale):
    """
    Generates a noisy signal.
    """

    N = noise.shape[0]

    mask_list = []
    for cov in coverage:
        mask_list.append(create_mask3(N, count=int(cov), size=(N + 1) // 100,
                                      align_scale=align_scale))
    mask = np.array(mask_list).transpose()

    return (signal[:, None, None]
            + noise[:, None, None] * mask[:, :, None] * SNR[None, None, :])
