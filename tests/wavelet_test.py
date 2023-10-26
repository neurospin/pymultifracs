import numpy as np

from pymultifracs.wavelet import wavelet_analysis
from pymultifracs.estimation import estimate_hmin


def test_wavelet_fbm(fbm_file):

    for fname in fbm_file:

        with open(fname, 'rb') as f:
            X = np.load(f)

        j2 = 7
        wt_coefs, _, _ = wavelet_analysis(X, p_exp=None, j2=j2)
        hmin = estimate_hmin(wt_coefs, [(2, wt_coefs.j2_eff())], weighted=None)[0]
        hmin = hmin.min()
        gamint = 0.0 if hmin >= 0 else 1
        wavelet_analysis(X, p_exp=np.inf, j2=j2)
        wavelet_analysis(X, p_exp=2, j2=j2, gamint=gamint)


def test_wavelet_mrw(mrw_file):

    for fname in mrw_file:

        with open(fname, 'rb') as f:
            X = np.load(f)

        j2 = 8
        wt_coefs, _, _ = wavelet_analysis(X, p_exp=None, j2=j2)
        hmin = estimate_hmin(wt_coefs, [(2, wt_coefs.j2_eff())], weighted=None)[0]
        hmin = hmin.min()
        gamint = 0.0 if hmin >= 0 else 1
        wavelet_analysis(X, p_exp=np.inf, j2=j2)
        wavelet_analysis(X, p_exp=2, j2=j2, gamint=gamint)
