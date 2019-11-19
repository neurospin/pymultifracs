from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt

from .mfa import MFA


def plot_psd(signal, fs, n_fft=4096, segment_size=None, n_moments=2):
    """
    Plot the superposition of Fourier-based Welch estimation and Wavelet-based
    estimation of PSD on a log-log graphic.
    Based on the `wavelet_fourier_spectrum` function of the MATLAB toolbox
    mf_bs_toolbox-v0.2 found at
    http://www.ens-lyon.fr/PHYSIQUE/Equipe3/Multifractal/dat/mf_bs_tool-v0.2.zip

    Parameters
    ----------
    signal: 1D-array_like
        Time series of sampled values

    fs: float
        Sampling frequency of the signal

    n_fft: int, optional
        Length of the FFT desired.
        If `segment_size` is greater, ``n_fft = segment_size``.

    segment_size: int | None
        Length of Welch segments.
        Defaults to None, which sets it equal to `n_fft`

    n_moments: int
        Number of vanishing moments of the Daubechies wavelet used in the
        Wavelet decomposition.

    """

    if segment_size is None:
        segment_size = n_fft

    if n_fft < segment_size:
        n_fft = segment_size

    # Fourier / Welch part

    _,psd_Fourier = welch(signal,
                          window='hamming',
                          nperseg=segment_size,
                          noverlap=segment_size/2,
                          nfft=n_fft,
                          detrend=False,
                          return_onesided=True,
                          scaling='density',
                          average='mean',
                          fs=2 * np.pi)

    freq_Fourier = fs * np.linspace(0, 0.5, n_fft / 2 + 1)
    freq_Fourier = np.log2(freq_Fourier[1:])
    psd_Fourier = np.log2(psd_Fourier[1:]) + 2
    # +2 is to compensate for negative frequencies

    # Wavelet part

    mfa = MFA(wt_name=f'db{n_moments}',
              formalism='wcmf',
              j1=1,
              j2=13,
              n_cumul=2,
              gamint=1/2,
              wtype=0,
              p=np.inf)

    mfa._set_and_verify_parameters()

    mfa.wavelet_coeffs = None
    mfa.wavelet_leaders = None
    mfa.structure = None
    mfa.cumulants = None

    mfa._wavelet_analysis(signal)

    psd_wavelet = [np.square(arr).mean() for arr in mfa.wavelet_coeffs.values.values()]

    j_wavelet = np.arange(len(psd_wavelet)) + 1
    freq_wavelet = (3/4 * fs) / (np.power(2, j_wavelet))
    freq_wavelet, psd_wavelet = np.log2(freq_wavelet), np.log2(psd_wavelet)

    # Plotting

    plt.plot(freq_Fourier, psd_Fourier)
    plt.plot(freq_wavelet, psd_wavelet)
    plt.legend(['Fourier', 'Wavelet'])
    plt.xlabel('log_2 f')
    plt.ylabel('log_2 S(f)')
    plt.title('Power Spectral Density')
    plt.show()
