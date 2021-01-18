"""
Authors: Merlin Dumeur <merlin@dumeur.net>
"""

from collections import namedtuple

from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt

from .wavelet import wavelet_analysis

PSD = namedtuple('PSD', 'freq psd')
"""Aggregates power spectral density information

Attributes
----------
freq : ndarray
    Frequency support of the psd values
psd : ndarray
    Power density associated matching the frequencies in ``freq``
"""


def plot_psd(signal, fs, n_fft=4096, seg_size=None, n_moments=2,
             log='log2'):
    """
    Plot the superposition of Fourier-based Welch estimation and Wavelet-based
    estimation of PSD on a log-log graphic.

    Based on the `wavelet_fourier_spectrum` function of the MATLAB toolbox
    mf_bs_toolbox-v0.2 [1]_

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Time series of sampled values

    fs : float
        Sampling frequency of ``signal``

    n_fft : int, optional
        Length of the FFT desired.
        If `seg_size` is greater, ``n_fft = seg_size``.

    seg_size : int | None
        Length of Welch segments.
        Defaults to None, which sets it equal to ``n_fft``

    n_moments : int
        Number of vanishing moments of the Daubechies wavelet used in the
        Wavelet decomposition.

    log : str
        Log function to use on the frequency and power axes

    References
    ----------
    .. [1] \
        http://www.ens-lyon.fr/PHYSIQUE/Equipe3/Multifractal/dat/mf_bs_tool-v0.2.zip
    """

    # Computing

    freq_fourier, psd_fourier = welch_estimation(signal, fs, n_fft,
                                                 seg_size)
    freq_wavelet, psd_wavelet = wavelet_estimation(signal, fs, n_moments)

    # Plotting

    freq = [freq_fourier, freq_wavelet]
    psd = [psd_fourier, psd_wavelet]
    legend = ['Fourier', 'Wavelet']
    log_plot(freq, psd, legend, log=log)


log_function = {'log2': np.log2,
                'log': np.log}


def _log_psd(freq, psd, log):
    """
    Compute the logged values of a PSD and its frequency support similarly to
    the MATLAB toolbox
    """

    # Avoid computing log(0)
    if np.any(freq == 0):
        support = [freq != 0.0][0]
        freq, psd = freq[support], psd[support]

    # Compute the logged values of the frequency and psd
    log = log_function[log]
    freq, psd = log(freq), log(psd)

    return freq, psd


def log_plot(freq_list, psd_list, legend=None, fmt=None, color=None, slope=[],
             log='log2', lowpass_freq=np.inf, xticks=None,
             title='Power Spectral Density', ax=None, show=False,
             plot_kwargs={}, linewidth=None):
    """
    Perform a log-log plot over a list of paired frequency range and PSD, with
    optional legend and fitted slope

    Parameters
    ----------
    freq_list: list
        list of frequency supports of the PSDs to plot

    psd_list: list
        list of PSDs to plot

    legend: list | None
        list of labels to assign to the PSDs

    color: list | None
        colors to assign to the plotted PSDs

    slope: [(freq, psd)] | []
        list of 2-tuples containing the frequency support and PSD
        representation of a slope to plot
        TODO: replace (freq, psd) with (beta, log_C)

    log: str
        name of log function to use on the data before plotting
    """

    ax = plt.gca() if ax is None else ax

    ax.set_xlabel(f'{log} f')
    ax.set_ylabel(f'{log} S_2(f)')
    ax.set_title(title)

    if color is None:
        cmap = plt.get_cmap("tab10")
        color = [cmap(i % 10) for i in range(len(freq_list))]

    if linewidth is None:
        linewidth = [1.5] * len(freq_list)

    if fmt is None:
        fmt = ['-'] * len(freq_list)

    for i, (freq, psd, f, col, lw) in enumerate(zip(freq_list, psd_list, fmt,
                                                    color, linewidth)):

        indx = tuple([freq < lowpass_freq])
        freq, psd = freq[indx], psd[indx]
        log_freq, psd = _log_psd(freq, psd, log)  # Log frequency and psd

        ax.plot(log_freq, psd, f, c=col, lw=lw, **plot_kwargs)

        if xticks is not None and i == len(freq_list) - 1:
            ax.set_xticks(log_freq)
            ax.set_xticklabels([f'{fr:.2f}' if fr < 1 else f'{fr:.1f}'
                                for fr in freq])
            # plt.xticks(log_freq, [f'{fr:.2f}' for fr in freq])

    if color is None:
        c_gen = (f'C{i}' for i in range(len(freq_list)))
    else:
        c_gen = color

    for tup, c in zip(slope, c_gen):
        ax.plot(*tup, color=c, dashes=[6, 2], lw=2)

    if legend is not None:
        ax.legend(legend)

    if show:
        plt.show()


def welch_estimation(signal, fs, n_fft=4096, seg_size=None):
    """
    Wrapper for :obj:`scipy.signal.welch`

    Parameters
    ----------
    signal : 1D-array_like
        Time series of sampled values

    fs : float
        Sampling frequency of the signal

    n_fft : int, optional
        Length of the FFT desired.
        If `seg_size` is greater, ``n_fft = seg_size``.

    seg_size : int | None
        Length of Welch segments.
        Defaults to None, which sets it equal to `n_fft`

    Returns
    -------
    psd : PSD
    """

    # Input argument sanitizing

    if seg_size is None:
        seg_size = n_fft

    if n_fft < seg_size:
        n_fft = seg_size

    # Frequency
    freq = fs * np.linspace(0, 0.5, n_fft // 2 + 1)

    # PSD
    _, psd = welch(signal,
                   window='hamming',
                   nperseg=seg_size,
                   noverlap=seg_size / 2,
                   nfft=n_fft,
                   detrend=False,
                   return_onesided=True,
                   scaling='density',
                   average='mean',
                   fs=2 * np.pi)

    psd *= 4        # compensating for negative frequencies
    psd = np.array(psd)

    return PSD(freq=freq, psd=psd)


def wavelet_estimation(signal, fs, n_moments, j2=None):
    """
    PSD estimation based on Wavelet coefficients

    The PSD is estimated using :obj:`~pymultifracs.wavelet.wavelet_analysis`.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Time series
    fs : float
        Sampling frequency of the signal
    n_moments : int
        Number of vanishing moments of the Daubechies wavelet used in the
        wavelet transform.

    Returns
    -------
    psd : PSD
    """

    # PSD
    transform = wavelet_analysis(signal, j1=1, j2=j2,
                                 normalization=1,
                                 wt_name=f'db{n_moments}',
                                 gamint=0.5,
                                 weighted=False,
                                 p_exp=None)

    psd = [np.square(arr).mean() for arr in transform.wt_coefs.values.values()]
    psd = np.array(psd)

    # Frequency
    scale = np.arange(len(psd)) + 1
    freq = (3/4 * fs) / (np.power(2, scale))

    return PSD(freq=freq, psd=psd)
