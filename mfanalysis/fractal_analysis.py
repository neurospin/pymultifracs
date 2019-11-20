from .psd import welch_estimation, wavelet_estimation, _log_plot, _log_psd

from sklearn.linear_model import Ridge


def plot_fractal(signal, fs, log='log2', cutoff_frequency=8, n_moments=2,
                 n_fft=4096, segment_size=None):
    """
    Plot the superposition of Welch and Wavelet-based estimation of PSD, along
    with the estimation of the $beta$ coefficient on a log-log graphic.

    Parameters
    ----------
    signal: 1D-array_like
        Time series of sampled values

    fs: float
        Sampling frequency of the signal

    log: str
        Log function to use on the PSD

    cutoff_frequency: int, optional
        Frequency (in Hertz) which delimitates the higher bound of frequencies
        to use during the estimation of $beta$

    n_moments: int, optional
        Number of vanishing moments of the Daubechies wavelet used in the
        Wavelet decomposition.

    n_fft: int, optional
        Length of the FFT desired.
        If `segment_size` is greater, ``n_fft = segment_size``.

    segment_size: int | None
        Length of Welch segments.
        Defaults to None, which sets it equal to `n_fft`

    """

    # Compute the PSD
    freq_fourier, psd_fourier = welch_estimation(signal, fs, n_fft,
                                                 segment_size)
    freq_wavelet, psd_wavelet = wavelet_estimation(signal, fs, n_moments)

    # Estimate the 1/f slope
    beta, log_C, freq_slope = estimate_beta(freq_wavelet, psd_wavelet,
                                         log, cutoff_frequency)

    # Compute values to plot the 1/f slope
    psd_slope = beta * freq_slope + log_C

    # Plot
    freq = [freq_fourier, freq_wavelet]
    psd = [psd_fourier, psd_wavelet]
    legend = ['Fourier', 'Wavelet', f'Slope: {beta:.2f}']

    _log_plot(freq, psd, legend, slope=(freq_slope, psd_slope), log=log)


# def fractal_analysis(signal, fs, n_moments=2, cutoff_frequency=8, log='log2'):

#     freq, psd = wavelet_estimation(signal, fs, n_moments)
#     beta, log_C, _ = estimate_beta(freq, psd, cutoff_frequency, log)

#     return beta, log_C


def estimate_beta(freq, psd, log='log2', cutoff_frequency=8):

    """
    Estimate the value of beta

    Parameters
    ----------
    freq: 1D-array_like
        Frequencies at which the PSD has been estimated

    psd: 1D-array_like
        Power spectral density

    log: str
        Log function to use on the PSD before fitting the slope

    cutoff_frequency: int, optional
        Frequency (in Hertz) which delimitates the higher bound of frequencies
        to use during the estimation of $beta$

    """

    # Low-pass the PSD
    support = [freq < cutoff_frequency][0]
    freq = freq[support].reshape(-1, 1)
    psd = psd[support]

    # Log the values
    freq, psd = _log_psd(freq, psd, log)

    # Fit ridge regressor
    R = Ridge()
    R.fit(freq, psd)

    return R.coef_[0], R.intercept_, freq
