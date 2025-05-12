"""
Authors: Merlin Dumeur <merlin@dumeur.net>
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .fbm import fbm
from .mrw import mrw
from .. import mfa, wavelet_analysis


# def create_mask2(length, count, size):

#     size = int(size)
#     mask = np.zeros(length, dtype=bool)

#     # for i in range(count):
#     #     start = rng.integers(pad, length - size - pad)
#     #     mask[start:start+size] = 1

#     points = np.linspace(0, length, count+2, dtype=int)

#     # print(points, size)

#     for midpoint in points[1:-1]:
#         # print(midpoint)
#         mask[midpoint - size // 2:midpoint + size // 2] = 1

#     # import matplotlib.pyplot as plt
#     # plt.plot(mask)

#     return mask


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


# def create_oscillation(length, freq, sampling_freq):

#     t = np.linspace(0, length / sampling_freq, length)
#     return np.exp(((t * freq / 2 / np.pi) % (2 * np.pi)) * 1j)


# def get_mask_durations(mask):

#     idx_start = np.arange(mask.shape[0]-1)[np.diff(mask) == 1]
#     idx_end = np.arange(mask.shape[0]-1)[np.diff(mask) == -1]

#     if len(idx_end) == len(idx_start) - 1:
#         idx_end = np.r_[idx_end, mask.shape[0]]
#     print((idx_end > idx_start).all())

#     return idx_end - idx_start


# def generate_osc_freq(length, H=.45, spread=.005, mean=.49, fs=1):

#     freq = fbm(shape=length, H=H)

#     freq -= freq.mean()
#     freq /= np.abs(freq.max())
#     freq *= spread

#     b, a = butter(4, .0001, fs=1, btype='low')
#     freq = lfilter(b, a, freq)
#     freq += mean

#     return freq


# def generate_simuls(N):

#     X = fbm(shape=N, H=0.8)
#     X_diff = np.diff(X)

#     freq = generate_osc_freq(N -1 +10000, spread=.01, mean=.49)
#     osc = create_oscillation(N-1, freq[10000:], 1)
#     osc *= X_diff.std()

#     return X_diff


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
