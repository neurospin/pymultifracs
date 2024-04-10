import numpy as np

from pymultifracs.wavelet import wavelet_analysis
from pymultifracs.estimation import estimate_hmin


def test_wavelet_fbm(fbm_file):

    for fname in fbm_file:

        with open(fname, 'rb') as f:
            X = np.load(f)

        WT = wavelet_analysis(X)
        WTL = WT.get_leaders(np.inf).auto_integrate([(2, WT.j2_eff())])


def test_wavelet_mrw(mrw_file):

    for fname in mrw_file:

        with open(fname, 'rb') as f:
            X = np.load(f)

        WT = wavelet_analysis(X)
        WTpL = WT.get_leaders(2).auto_integrate([(2, WT.j2_eff())])
