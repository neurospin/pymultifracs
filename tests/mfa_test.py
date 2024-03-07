import pytest

import numpy as np
import json

from pymultifracs.wavelet import wavelet_analysis
from pymultifracs.estimation import estimate_hmin
from pymultifracs.mfa import mf_analysis


@pytest.mark.mfa
def test_mfa_fbm(fbm_file):

    with open('tests/fbm_config.json', 'rb') as f:
        config_list = json.load(f)

    for i, fname in enumerate(fbm_file):

        with open(fname, 'rb') as f:
            X = np.load(f)

        j2 = int(np.log2(config_list[i]['shape']) - 3)
        wt_coefs, wt_leaders, _ = wavelet_analysis(
            X, p_exp=2, j2=j2)

        scaling_ranges = [(3, wt_leaders.j2_eff())]

        hmin = estimate_hmin(wt_coefs, scaling_ranges, weighted='Nj')[0]
        hmin = estimate_hmin(wt_coefs, scaling_ranges, weighted=None)[0]
        hmin = hmin.min()
        gamint = 0.0 if hmin >= 0 else 1

        q = np.array([-2, -1, 0, 1, 2])

        dwt, lwt = mf_analysis_full(X, scaling_ranges, gamint=gamint,
                                    p_exp=np.inf, n_cumul=3, q=q)
        if config_list[i]['H'] != 0.01:
            assert abs(dwt.structure.H.mean() - gamint - config_list[i]['H']) < 0.1,\
                print(f'{dwt.structure.H.mean()=}, {config_list[i]["H"]=}, '
                      f'{gamint=}')
        assert abs(lwt.cumulants.log_cumulants[1, :].mean()) < 0.01

        _, lwt = mf_analysis_full(X, scaling_ranges, gamint=gamint, p_exp=2,
                                  n_cumul=3, q=q)
        assert abs(lwt.cumulants.log_cumulants[1, :].mean()) < 0.01


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
        wt_coefs, wt_leaders, _ = wavelet_analysis(X, p_exp=2, j2=j2)

        scaling_ranges = [(3, wt_leaders.j2_eff())]

        hmin = estimate_hmin(wt_coefs, scaling_ranges, weighted='Nj')[0]
        hmin = estimate_hmin(wt_coefs, scaling_ranges, weighted=None)[0]
        hmin = hmin.min()
        gamint = 0.0 if hmin >= 0 else 1

        q = np.array([-2, -1, 0, 1, 2])

        dwt, lwt = mf_analysis_full(X, scaling_ranges, gamint=gamint,
                                    p_exp=np.inf, n_cumul=3, q=q)
        assert abs(dwt.structure.H.mean() - gamint - config_list[i]['H']) < 0.1
        assert abs(lwt.cumulants.c2.mean()
                   + (config_list[i]['lam'] ** 2)) < 0.025

        _, lwt = mf_analysis_full(X, scaling_ranges, gamint=gamint, p_exp=2,
                                  n_cumul=3, q=q)
        assert abs(lwt.cumulants.c2.mean()
                   + (config_list[i]['lam'] ** 2)) < 0.025
