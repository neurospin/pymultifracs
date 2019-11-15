from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt

from .mfa import MFA

def plot_psd(signal, fs):

    overlapingratio = 4
    window_size = np.power(2, np.fix(np.log2(len(signal))) - overlapingratio)

    _,pds_Fourier = welch(signal,
                    window='hamming',
                    nperseg=window_size,
                    noverlap=window_size/2,
                    nfft=4096,
                    detrend=False,
                    return_onesided=True,
                    scaling='density',
                    average='mean',
                    fs = 2*np.pi)

    f_Fourier = fs * np.linspace(0,0.5, window_size / 2 + 1)
    f_Fourier, pds_Fourier = np.log2(f_Fourier[1:]), np.log2(pds_Fourier[1:]) + 2

    mfa = MFA(wt_name=f'db2',
                 formalism='wcmf',
                 j1=1,j2=13,
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

    pds_wavelet = [np.square(arr).mean() for arr in mfa.wavelet_coeffs.values.values()]
    
    j_wavelet = np.arange(len(pds_wavelet))+1
    f_wavelet = (3/4 * fs) / (np.power(2, j_wavelet))
    
    f_wavelet, pds_wavelet = np.log2(f_wavelet), np.log2(pds_wavelet)

    plt.plot(f_Fourier, pds_Fourier)
    plt.plot(f_wavelet, pds_wavelet)
    plt.legend(['Fourier', 'Wavelet'])
    plt.xlabel('log_2 f')
    plt.ylabel('log_2 S(f)')
    plt.title('Power Spectral Density log-log plot')
    plt.show()

    plt.show()