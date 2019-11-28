from dataclasses import dataclass
from collections import namedtuple

import numpy as np

from .fractal_analysis import plot_fractal, estimate_beta, FractalValues
from .psd import plot_psd, wavelet_estimation
from .wavelet import wavelet_analysis
from .mf_analysis import mf_analysis
from .estimation import compute_hurst, estimate_hmin


WaveletParameters = namedtuple('WaveletParameters', ['j1',
                                                     'j2',
                                                     'normalization',
                                                     'gamint',
                                                     'weighted',
                                                     'wt_name',
                                                     'p_exp'])

WaveletPSD = namedtuple('WaveletPSD', 'freq psd')

FractalParameters = namedtuple('FractalParameters', ['n_moments',
                                                     'cutoff_freq'])


@dataclass
class Signal:
    """
    Class used to manage the signal data, in order to reuse intermediary
    results, and share global parameters.
    """
    data: np.ndarray
    fs: float
    log: str = 'log2'
    wt_psd_moments: int = None
    wt_psd: WaveletPSD = None
    fractal_param: FractalParameters = None
    fractal: FractalValues = None
    wt_transform = WaveletTransform = None
    wt_param: WaveletParameters = None

    def _wavelet_psd(self, n_moments=2):

        if self.wt_psd is None or self.wt_psd_moments != n_moments:

            self.wt_psd_moments = n_moments

            freq, psd = wavelet_estimation(self.data, self.fs,
                                           n_moments=n_moments)

            self.wt_psd = WaveletPSD(freq=freq,
                                     psd=psd)

    def plot_psd(self, n_fft=4096, segment_size=None, n_moments=2):

        plot_psd(self.data, self.fs, log=self.log,
                 n_moments=n_moments,
                 segment_size=segment_size,
                 n_fft=n_fft)

    def plot_fractal(self, n_moments=2, cutoff_freq=8,
                     n_fft=4096, segment_size=None):

        plot_fractal(self.data, self.fs, log=self.log,
                     segment_size=segment_size,
                     cutoff_freq=cutoff_freq,
                     n_moments=n_moments,
                     n_fft=n_fft)

    def fractal_analysis(self, n_moments=2, cutoff_freq=8):

        self._wavelet_psd(n_moments)
        self.fractal = estimate_beta(self.wt_psd.freq,
                                     self.wt_psd.psd,
                                     log=self.log,
                                     cutoff_freq=cutoff_freq)

        return self.fractal

    def _wavelet_analysis(self, new_param):

        if self.wt_param is None or self.wt_param != new_param:
            self.wt_param = new_param

        # self.wt_transform = wavelet_analysis(self.data, j1, j2, normalization,
                                            #  gamint, weighted, wt_name, p_exp)
        self.wt_transform = wavelet_analysis(self.data,
                                             **self.wt_param._asdict())

        return self.wt_transform

    def wavelet_analysis(self, j1, j2, normalization, gamint, weighted,
                         wt_name, p_exp):

        new_param = WaveletParameters(j1=j1,
                                      j2=j2,
                                      normalization=normalization,
                                      gamint=gamint,
                                      weighted=weighted,
                                      wt_name=wt_name,
                                      p_exp=p_exp)

        return self._wavelet_analysis(new_param)

    def _check_wt_transform(self):

        if self.wt_transform is None or self.wt_param is None:
            raise AttributeError("Wavelet analysis was not done")

    def mf_analysis(self, q, n_cumul):

        self._check_wt_transform()

        return mf_analysis(**self.wt_transform._asdict(),
                           p_exp=self.wt_param.p_exp,
                           j1=self.wt_param.j1,
                           weighted=self.wt_param.weighted,
                           q=q,
                           n_cumul=n_cumul)

    def hurst(self):

        self._check_wt_transform()

        return compute_hurst(self.wt_transform.wt_coefs,
                             self.wt_param.j1, self.wt_transform.j2_eff,
                             self.wt_param.weighted)

    def hmin(self):

        self._check_wt_transform()

        return estimate_hmin(self.wt_transform.wt_coefs,
                             j1=self.wt_param.j1,
                             j2_eff=self.wt_transform.j2_eff,
                             weighted=self.wt_param.weighted)

    def mf_analysis_full(self, j1=1, j2=10, normalization=1, gamint=0.0,
                         weighted=True, wt_name='db3', p_exp=None, q=None,
                         n_cumul=3):

        if q is None:
            q = [2]

        self.wavelet_analysis(j1, j2, normalization, gamint, weighted, wt_name,
                              p_exp)

        return self.mf_analysis(q, n_cumul)
