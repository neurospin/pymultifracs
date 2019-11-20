from dataclasses import dataclass
import numpy as np

from .fractal_analysis import plot_fractal, estimate_beta
from .psd import plot_psd, welch_estimation, wavelet_estimation


@dataclass
class Signal:
    data: np.ndarray
    fs: float
    log: str = 'log2'

    def plot_fractal(self, n_moments=2, cutoff_frequency=8,
                     n_fft=4096, segment_size=None):

        plot_fractal(self.data, self.fs, log=self.log,
                     cutoff_frequency=cutoff_frequency,
                     segment_size=segment_size,
                     n_moments=n_moments,
                     n_fft=n_fft)

    def fractal_analysis(self, n_moments=2, cutoff_frequency=8):

        freq, psd = self._wavelet_estimation(n_moments)
        beta, log_C, _ = estimate_beta(freq, psd, log=self.log,
                                       cutoff_frequency=cutoff_frequency)

        return beta, log_C

    def plot_psd(self, n_fft=4096, segment_size=None, n_moments=2):

        plot_psd(self.data, self.fs, log=self.log,
                 n_moments=n_moments,
                 segment_size=None,
                 n_fft=n_fft)

    # def welch_estimation(self, n_fft=4096, segment_size=None):

    #     return welch_estimation(self.data, self.fs, log=self.log,
    #                             segment_size=segment_size,
    #                             n_fft=n_fft)

    def _wavelet_estimation(self, n_moments=2):

        return wavelet_estimation(self.data, self.fs,
                                  n_moments=n_moments)
