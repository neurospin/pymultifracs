import pytest

import numpy as np
import json

from pymultifracs.estimation import estimate_hmin
from pymultifracs import mfa, wavelet_analysis
from pymultifracs.utils import Dim


@pytest.mark.mfa
def test_mfa_fbm(fbm_file):

    with open('tests/fbm_config.json', 'rb') as f:
        config_list = json.load(f)

    for i, fname in enumerate(fbm_file):

        with open(fname, 'rb') as f:
            X = np.load(f)

        j2 = int(np.log2(config_list[i]['shape']) - 3)

        WTL = wavelet_analysis(X, j2=j2).get_leaders(p_exp=2)

        scaling_ranges = [(3, WTL.j2_eff())]

        WTL = WTL.auto_integrate(scaling_ranges)

        assert WTL.gamint == 0

        q = np.array([-2, -1, 0, 1, 2])

        dwt, lwt = mfa([WTL.origin_mrq, WTL], scaling_ranges, n_cumul=4, q=q)

        lwt.structure.get_jrange(1, 2, False)

        if config_list[i]['H'] != 0.01:
            assert abs(dwt.structure.H.mean(dim=Dim.channel) - WTL.gamint - config_list[i]['H']) < 0.11
        assert abs(lwt.cumulants.log_cumulants.sel(m=2).mean(dim=Dim.channel)) < 0.0105


@pytest.mark.mfa
def test_mfa_mrw(mrw_file):

    with open('tests/mrw_config.json', 'rb') as f:
        config_list = json.load(f)

    for i, fname in enumerate(mrw_file):

        with open(fname, 'rb') as f:
            X = np.load(f)

        if config_list[i]['H'] == 0.01:
            continue

        j2 = int(np.log2(X.shape[0]) - 3)

        WT = wavelet_analysis(X, j2=j2)
        WTpL = WT.get_leaders(p_exp=2)

        scaling_ranges = [(3, WTpL.j2_eff())]

        WTpL = WTpL.auto_integrate(scaling_ranges)
        WT = WT.auto_integrate(scaling_ranges)

        assert WTpL.gamint == 0

        q = np.array([-2, -1, 0, 1, 2])

        dwt, lwt = mfa(
            [WT, WTpL], scaling_ranges, n_cumul=4, q=q)
        assert abs(dwt.structure.H.mean(dim=Dim.channel) - WT.gamint - config_list[i]['H']) < 0.11
        assert abs(lwt.cumulants.c2.mean(dim=Dim.channel)
                   + (config_list[i]['lam'] ** 2)) < 0.025


@pytest.mark.mfa
def test_goodness_fit(mrw_file):

    with open(mrw_file[0], 'rb') as f:
        X = np.load(f)

    WTpL = wavelet_analysis(X).integrate(1).get_leaders(p_exp=2)

    scaling_ranges = [(3, WTpL.j2_eff())]

    pwt = mfa(WTpL, scaling_ranges, n_cumul=2)

    pwt.cumulants.compute_R2()
    pwt.cumulants.compute_RMSE()
