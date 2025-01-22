import pytest
import json
import warnings

import numpy as np

from pymultifracs.estimation import estimate_hmin
from pymultifracs import mfa, wavelet_analysis
from pymultifracs.robust import get_outliers
from pymultifracs.simul.noisy import create_mask3
from pymultifracs.simul import fbm

@pytest.mark.robust
def test_unit_outlier_fbm(fbm_file):

    with open('tests/fbm_config.json', 'rb') as f:
        config_list = json.load(f)

    fname = fbm_file[0]

    with open(fname, 'rb') as f:
        X = np.diff(np.load(f)[:, 0])

    N = X.shape[0]
    j2 = int(np.log2(N) - 3)
    coverage = 3
    align_scale = j2 - 1

    mask = create_mask3(
        N, count=int(coverage), size=(N) // 100,
        align_scale=align_scale)
    
    noise = np.diff(fbm(shape=N+1, H=.5), axis=0)[:, None]
    noisy_X = X[:, None] + noise * mask[:, None] * 8

    scaling_ranges = [(3, j2-2)]
    pelt_beta = 5
    threshold = 3
    pelt_jump = 16

    WT = wavelet_analysis(noisy_X)

    _, idx_reject = get_outliers(
        WT, scaling_ranges, pelt_beta, threshold, pelt_jump,
        generalized=False)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
    
        _, idx_reject = get_outliers(
            WT, scaling_ranges, pelt_beta, threshold, pelt_jump,
            generalized=True)

    WT = WT.integrate(1).get_leaders(2)

    lwt = mfa(
        WT, scaling_ranges=scaling_ranges, robust=False, weighted=None,
        n_cumul=2, idx_reject=idx_reject)
    
    WT.plot(3, j2-2, nan_idx=idx_reject)


@pytest.mark.robust
def test_unit_robust_cm(fbm_file):

    with open('tests/fbm_config.json', 'rb') as f:
        config_list = json.load(f)

    fname = fbm_file[0]

    with open(fname, 'rb') as f:
        X = np.diff(np.load(f)[:, 0])

    WTpL = wavelet_analysis(X).integrate(1).get_leaders(2)

    scaling_ranges = [(3, 7)]

    lwt = mfa(
        WTpL, scaling_ranges=scaling_ranges, robust=True, weighted=None,
        n_cumul=2, idx_reject=None)
