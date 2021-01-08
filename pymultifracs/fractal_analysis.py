from collections import namedtuple

import numpy as np
from sklearn.linear_model import LinearRegression

from .psd import welch_estimation, wavelet_estimation, log_plot, _log_psd


def plot_fractal(signal, s_freq, log='log2', freq_band=(0.01, 2), n_moments=2,
                 n_fft=4096, seg_size=None, lowpass_freq=49):
    """
    Plot the superposition of Welch and Wavelet-based estimation of PSD, along
    with the estimation of the $beta$ coefficient on a log-log graphic.

    Parameters
    ----------
    signal: 1D-array_like
        Time series to process

    s_freq: float
        Sampling frequency of the signal

    log: str
        Log function to use on the PSD

    freq_band: int, optional
        Frequency (in Hertz) which delimitates the higher bound of frequencies
        to use during the estimation of $beta$

    n_moments: int, optional
        Number of vanishing moments of the Daubechies wavelet used in the
        Wavelet decomposition.

    n_fft: int, optional
        Length of the FFT desired.
        If `seg_size` is greater, ``n_fft = seg_size``.

    seg_size: int | None
        Length of Welch segments.
        Defaults to None, which sets it equal to `n_fft`

    """

    # Compute the PSD
    freq_fourier, psd_fourier = welch_estimation(signal, s_freq, n_fft,
                                                 seg_size)
    freq_wavelet, psd_wavelet = wavelet_estimation(signal, s_freq, n_moments)

    # Estimate the 1/f slope
    slope = estimate_beta(freq_wavelet, psd_wavelet, freq_band, log)

    # Compute values to plot the 1/f slope
    psd_slope = slope.beta * slope.freq + slope.log_C

    # Plot
    freq = [freq_fourier, freq_wavelet]
    psd = [psd_fourier, psd_wavelet]
    legend = ['Fourier', 'Wavelet', f'Slope: {slope.beta:.2f}']

    log_plot(freq, psd, legend, slope=[(slope.freq, psd_slope)], log=log, lowpass_freq=lowpass_freq)


def fractal_analysis(signal, s_freq, n_moments=2, freq_band=(0.01, 2), log='log2'):
    """
    Perform the estimation of the value of beta and the logged value of C, \
    where beta and log_C are defined in relation with the PSD:
        PSD(f) = C|f|^-beta

    signal: 1D-array_like
        Time series to process

    s_freq: float
        Sampling frequency of the signal

    n_moments: int, optional
        Number of vanishing moments of the Daubechies wavelet used in the
        Wavelet decomposition.

    freq_band: int, optional
        Frequency (in Hertz) which delimitates the higher bound of frequencies
        to use during the estimation of `beta`

    log: str
        Log function to apply to the PSD

    """

    freq, psd = wavelet_estimation(signal, s_freq, n_moments)
    fractal = estimate_beta(freq, psd, freq_band, log)

    return fractal


FractalValues = namedtuple('FractalValues', ['beta',
                                             'log_C',
                                             'freq'])


def estimate_beta(freq, psd, freq_band=(0.01, 2), log='log2'):
    """
    From the PSD and its frequency support, estimate the C and beta variables, where
        PSD(f) = C|f|^-beta

    Parameters
    ----------
    freq: 1D-array_like
        Frequencies at which the PSD has been estimated

    psd: 1D-array_likes
        Power spectral density

    freq_band: int, optional
        Frequency (in Hertz) which delimitates the higher bound of frequencies
        to use during the estimation of $beta$

    log: str
        Log function to use on the PSD before fitting the slope

    """

    # Band-pass the PSD

    support = np.logical_and(freq_band[0] <= freq, freq <= freq_band[1])
    freq = freq[support].reshape(-1, 1)
    psd = psd[support]

# Move this input sanitization to the megfractal package instead
    # if psd.min() < 0:
    #     psd -= psd.min()
    #     psd += 1e-26

    # Log the values
    freq, psd = _log_psd(freq, psd, log)

    # Fit ridge regressor
    # regressor = Ridge()
    regressor = LinearRegression()

    regressor.fit(freq, psd)

    return FractalValues(beta=regressor.coef_[0],
                         log_C=regressor.intercept_,
                         freq=freq)
