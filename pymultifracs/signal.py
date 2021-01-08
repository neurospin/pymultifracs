from dataclasses import dataclass
from collections import namedtuple

import numpy as np

from .fractal_analysis import estimate_beta, FractalValues
from .psd import log_plot, wavelet_estimation, welch_estimation, PSD
from .wavelet import wavelet_analysis, WaveletTransform
from .mf_analysis import mf_analysis, MFractalData
from .estimation import compute_hurst, estimate_hmin


WaveletParameters = namedtuple('WaveletParameters', ['j1',
                                                     'j2',
                                                     'normalization',
                                                     'gamint',
                                                     'weighted',
                                                     'wt_name',
                                                     'p_exp'])

FractalParameters = namedtuple('FractalParameters', ['n_moments',
                                                     'freq_band'])

WelchParameters = namedtuple('WelchParameters', 'n_fft seg_size')

WTParametersPSD = namedtuple('WTParametersPSD', 'n_moments j2')

MFParameters = namedtuple('MFParameters', 'q n_cumul')


def same_params(old_params, new_params):

    for param in old_params._asdict():

        old_param = getattr(old_params, param)
        new_param = getattr(new_params, param)

        if (isinstance(old_param, np.ndarray)
           or isinstance(new_params, np.ndarray)):

            same = (old_param == new_param).all()

        else:

            same = old_param == new_param

        if not(same):
            return False

    return True


@dataclass
class Signal:
    """
    Class used to manage the signal data, in order to reuse intermediary
    results, and share global parameters.

    Attributes
    ----------

    data: np.ndarray
        Time series to process

    fs: float
        Sampling frequency of the signal

    log: str, optional
        name of the log function to use on the PSD

    wt_psd_moments: int | None
        number of vanishing moments of the Daubechies wavelet in the wavelet
            PSD estimation

    wt_psd: PSD | None
        stores the output of the wavelet PSD

    fractal_param: FractalParameters | None
        stores the parameters used in the fractal analysis

    fractal: FractalValues | None
        stores the output of the fractal analysis

    wt_transform: WaveletTransform | None
        stores the output of the wavelet transform

    wt_param: WaveletParameters | None
        stores the parameters used in the wavelet transform


    TODO add a method to fetch values of interest
    """
    data: np.ndarray
    fs: float
    log: str = 'log2'
    wt_psd_param: WTParametersPSD = None
    wt_psd: PSD = None
    welch_param: WelchParameters = None
    welch_psd: PSD = None
    fractal_param: FractalParameters = None
    fractal: FractalValues = None
    wt_transform: WaveletTransform = None
    wt_param: WaveletParameters = None
    mf_param: MFParameters = None
    multi_fractal: MFractalData = None

    def estimate_wavelet_psd(self, n_moments=2, j2=None):

        param = WTParametersPSD(n_moments=n_moments, j2=j2)

        if self.wt_psd is None or self.wt_psd_param != param:

            self.wt_param = param

            self.wt_psd = wavelet_estimation(self.data, self.fs,
                                             n_moments=n_moments,
                                             j2=j2)

        return self.wt_psd

    def estimate_welch_psd(self, n_fft=4096, seg_size=None):

        param = WelchParameters(n_fft=n_fft, seg_size=seg_size)

        if self.welch_psd is None or self.welch_param != param:

            self.welch_param = param

            self.welch_psd = welch_estimation(self.data, self.fs,
                                              n_fft=n_fft,
                                              seg_size=seg_size)

        return self.welch_psd

    def plot_psd(self, n_fft=4096, seg_size=None, n_moments=2, j2=None):

        self.estimate_wavelet_psd(n_moments, j2)
        self.estimate_welch_psd(n_fft, seg_size)

        log_plot([self.welch_psd.freq, self.wt_psd.freq],
                 [self.welch_psd.psd, self.wt_psd.psd],
                 ['Fourier', 'Wavelet'],
                 log=self.log)

    def plot_fractal(self, n_moments=2, freq_band=(0.01, 2),
                     n_fft=4096, seg_size=None):

        self.estimate_wavelet_psd(n_moments, None)
        self.estimate_welch_psd(n_fft, seg_size)
        self.fractal_analysis(n_moments, freq_band)

        psd_slope = self.fractal.beta * self.fractal.freq + self.fractal.log_C

        log_plot([self.welch_psd.freq, self.wt_psd.freq],
                 [self.welch_psd.psd, self.wt_psd.psd],
                 ['Fourier', 'Wavelet', f'Slope: {self.fractal.beta:.2f}'],
                 slope=[(self.fractal.freq, psd_slope)],
                 log=self.log)

    def fractal_analysis(self, n_moments=2, freq_band=(0.01, 2)):

        self.fractal_param = FractalParameters(n_moments, freq_band)
        self.estimate_wavelet_psd(n_moments, None)
        self.fractal = estimate_beta(self.wt_psd.freq,
                                     self.wt_psd.psd,
                                     log=self.log,
                                     freq_band=freq_band)

        return self.fractal

    def _wavelet_analysis(self, new_param):

        if self.wt_param is None or self.wt_param != new_param:

            self.wt_param = new_param
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

    def psd_difference(self, other, log=None):
        # the log argument is a pair of functions with the second one being
        # the inverse of the first

        common_support, idx_self, idx_other = np.intersect1d(
            self.wt_psd.freq, other.wt_psd.freq,
            assume_unique=True, return_indices=True)

        if log is None:
            psd = self.wt_psd.psd[idx_self] - other.wt_psd.psd[idx_other]
        else:
            self_psd = log[0](self.wt_psd.psd[idx_self])
            other_psd = log[0](other.wt_psd.psd[idx_other])

            psd = self_psd - other_psd
            psd -= psd.min()
            psd += 0.1

            psd = log[1](psd)

        return Signal(None, self.fs, self.log, self.wt_psd_param,
                      PSD(common_support, psd))

    def _check_wt_transform(self):

        if self.wt_transform is None or self.wt_param is None:
            raise AttributeError("Wavelet analysis was not done")

    def mf_analysis(self, q, n_cumul, save=True):

        self._check_wt_transform()

        new_param = MFParameters(q, n_cumul)

        if self.multi_fractal is None or not(same_params(self.mf_param,
                                                         new_param)):

            self.mf_param = new_param

            multi_fractal = mf_analysis(
                **self.wt_transform._asdict(),
                p_exp=self.wt_param.p_exp,
                j1=self.wt_param.j1,
                weighted=self.wt_param.weighted,
                q=q,
                n_cumul=n_cumul
            )

            if save:
                self.multi_fractal = multi_fractal

        return multi_fractal

    def hurst(self):
        # TODO store Hurst exponent

        self._check_wt_transform()

        return compute_hurst(self.wt_transform.wt_coefs,
                             self.wt_param.j1, self.wt_transform.j2_eff,
                             self.wt_param.weighted)

    def hmin(self):
        # TODO store h_min

        self._check_wt_transform()

        return estimate_hmin(self.wt_transform.wt_coefs,
                             j1=self.wt_param.j1,
                             j2_eff=self.wt_transform.j2_eff,
                             weighted=self.wt_param.weighted)

    def mf_analysis_full(self, j1=1, j2=10, normalization=1, gamint=0.0,
                         weighted=True, wt_name='db3', p_exp=None, q=None,
                         n_cumul=3, save=True):

        if q is None:
            q = [2]

        self.wavelet_analysis(j1, j2, normalization, gamint, weighted, wt_name,
                              p_exp)

        return self.mf_analysis(q, n_cumul, save)
