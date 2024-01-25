from typing import Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve, convolve, oaconvolve
import fcwt

from pymultifracs.simul import mrw


Formalism = Enum('Formalism', ['WT_COEF', 'WT_LEADER', 'WSE'])


def cwt(signal, sigma, fs, f0, f1, fn):

    morl = fcwt.Morlet(sigma)
    scales = fcwt.Scales(morl, fcwt.FCWT_LOGSCALES, fs, f0, f1, fn)

    fcwt_obj = fcwt.FCWT(morl, 1, False, True)
    cwt = np.zeros((fn, signal.shape[0]), dtype=np.complex64)
    fcwt_obj.cwt(signal, scales, cwt)

    freqs = np.zeros((fn,), dtype=np.float32)
    scales.getFrequencies(freqs)
    scales = fs / freqs

    return ContinuousMRQ(cwt.squeeze(), sigma, scales, fs, Formalism.WT_COEF)


def remove_edges_cwt(cwt):
    for i, s in enumerate(cwt.scales):
        support = int(np.ceil(3 * cwt.sigma * s))
        cwt.data[i, :support] = np.nan
        cwt.data[i, -support:] = np.nan


def remove_edges_leaders(leader, scales, sigma):
    for i, s in enumerate(scales):
        support = int(np.ceil(3 * sigma * s))
        leader[i, :support] = np.nan
        leader[i, -support:] = np.nan


@dataclass
class ContinuousMRQ:
    values: np.ndarray
    sigma: float
    scales: np.ndarray
    sfreq: float
    formalism: Formalism
    p_exp: Union[float, None] = None
    
    def compute_leaders(self):

        if self.formalism != 'wavelet coef':
            raise ValueError(
                'This method should be invoked from a wavelet coef object, '
                f'current object is {self.formalism}')

        leader = np.abs(self.values).squeeze()

        for i in range(1, self.scales.size):
            np.max(leader[i-1:i+1], axis=0, out=leader[i])

        for i, s in enumerate(self.scales):
            maximum_filter(leader[i], size=(2 * s,), mode='constant',
                           output=leader[i], cval=np.nan)

        remove_edges_leaders(leader, self.scales, self.sigma)

        return ContinuousMRQ(leader, self.sigma, self.scales, self.sfreq,
                             Formalism.WT_LEADER, np.inf)
    
    def compute_pleaders(self, p_exp):

        if p_exp == np.inf:
            return self.compute_leaders()
        
        if self.formalism != 'wavelet coef':
            raise ValueError(
                'This method should be invoked from a wavelet coef object, '
                f'current object is {self.formalism}')
        
        pleaders = np.abs(self.values.squeeze()) ** 2

        for i in range(1, self.scales.size):
            np.sum(pleaders[i-1:i+1], axis=0, out=pleaders[i])

        pleaders **= p_exp/2

        kernel = np.zeros(
            (pleaders.shape[0], int(np.floor(2 * self.scales[-1])))
            )

        for i, s in enumerate(self.scales):
            offset = int(np.floor((self.scales[-1] - s)))
            kernel[i, offset:offset+int(np.floor(2 * s))] = 1

        pleaders = oaconvolve(pleaders, kernel, axes=1, mode='same')
        pleaders[pleaders < 0] = 0

        pleaders /= self.scales[:, None]
        pleaders **= (1/p_exp)

        remove_edges_leaders(pleaders, self.scales, self.sigma)

        return ContinuousMRQ(pleaders, self.sigma, self.scales, self.sfreq,
                             Formalism.WT_LEADER, p_exp)

    def compute_wse(self, theta=.5):
   
        wse = np.zeros_like(self.values, dtype=np.float64)

        for idx_upper, scale in enumerate(self.scales):

            footprint, idx_lower = create_footprint(idx_upper, self.scales, theta)
            sl = np.abs(cwt.squeeze())[idx_lower:idx_upper+1]

            wse[idx_upper] = np.max(
                maximum_filter(sl, footprint=footprint,mode='constant',
                               cval=np.nan, axes=1),
                axis=0)

            max_width = width_from_scale(self.scales[idx_upper], self.scales[idx_lower])
            edge = int(np.ceil(3 * self.sigma * scale + 2 * max_width))
            wse[idx_upper, :edge] = np.nan
            wse[idx_upper, -edge:] = np.nan

        return ContinuousMRQ(wse, self.sigma, self.scales, self.sfreq,
                             Formalism.WSE)


def width_from_scale(a, scale):
    return int(2 ** (np.log2(a) - np.log2(scale) + 1)
               * (np.log2(a) - np.log2(scale) + 1))


def get_lower_scale_idx(scales, i, theta):

    lower_scale = max(1, 2 ** (np.log2(scales[i]) - np.log2(scales[i]) ** theta))
    return np.argmin(abs(lower_scale - scales))


def create_footprint(idx_upper, scales, theta):

    idx_lower = get_lower_scale_idx(scales, idx_upper, theta)
    kernel = np.zeros((idx_upper - idx_lower + 1))

    for kernel_idx, scale_idx in enumerate(range(idx_upper, idx_lower-1, -1)):
        kernel[-(kernel_idx+1)] = width_from_scale(scales[idx_upper], scales[scale_idx])

    return kernel, idx_lower
