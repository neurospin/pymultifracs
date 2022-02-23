import pytest
import json

import numpy as np

from pymultifracs.wavelet import wavelet_analysis
from pymultifracs.estimation import estimate_hmin
from pymultifracs.mfa import minimal_mf_analysis


@pytest.mark.bootstrap
def test_wavelet_bootstrap(mrw_file):

    for fname in mrw_file:

        with open(fname, 'rb') as f:
            X = np.load(f)

        j2 = 8
        wt_coefs, _, j2_eff, _ = wavelet_analysis(X, p_exp=None, j2=j2)
        hmin = estimate_hmin(wt_coefs, j1=1, j2_eff=j2_eff, weighted='Nj')[0]
        hmin = hmin.min()
        gamint = 0.0 if hmin >= 0 else -hmin + 0.1
        WT = wavelet_analysis(X, p_exp=2, j2=j2, gamint=gamint)
        coef_boot = WT.wt_coefs.bootstrap(5, 'db3')
        leader_boot = WT.wt_leaders.bootstrap(5, 'db3')


@pytest.mark.bootstrap
def test_confidence_interval(mrw_file):

    with open('tests/mrw_config.json', 'rb') as f:
        config_list = json.load(f)

    for i, fname in enumerate(mrw_file):

        with open(fname, 'rb') as f:
            X = np.load(f)

        if config_list[i]['H'] == 0.01:
            continue

        # j2 = int(np.log2(X.shape[0]) - 3)
        # wt_coefs, _, j2_eff, _ = wavelet_analysis(X, p_exp=None, j2=j2)

        # hmin = estimate_hmin(wt_coefs, j1=1, j2_eff=j2_eff, weighted=True)[0]
        # hmin = hmin.min()
        # gamint = 0.0 if hmin >= 0 else -hmin + 0.1

        j2 = int(np.log2(X.shape[0]) - 3)
        WT = wavelet_analysis(X, p_exp=2, j1=2, j2=12, weighted=None)

        coef_boot = WT.wt_coefs.bootstrap(5, 'db3')
        leader_boot = WT.wt_leaders.bootstrap(5, 'db3')

        gamint = 1
        j2_eff = coef_boot.j2_eff()

        q = np.array([2])
        scaling_ranges = [(3, j2_eff)]

        dwt, lwt = minimal_mf_analysis(
            WT.wt_coefs, WT.wt_leaders, j1=2, weighted='Nj', q=None,
            n_cumul=3)

        dwt_b, lwt_b = minimal_mf_analysis(
            coef_boot, leader_boot, j1=2, weighted='Nj', q=None, n_cumul=3)

        lwt_b.cumulants.CI_c2
        lwt_b.cumulants.CIE_c2(lwt.cumulants)
        lwt_b.cumulants.CI_C2
        lwt_b.cumulants.CIE_C2(lwt.cumulants)

        dwt_b.structure.CI_s_q(2)
        dwt_b.structure.CIE_s_q(dwt.structure)(2)
        dwt_b.structure.CI_S_q(2)
        dwt_b.structure.CIE_S_q(dwt.structure)(2)

        # minimal_mf_analysis(WT, scaling_ranges, gamint=gamint,p_exp=2, n_cumul=3, q=q)
        # assert abs(dwt.structure.H.mean() - config_list[i]['H']) < 0.1
        # assert abs(lwt.cumulants.log_cumulants[1, :].mean()
        #            + (config_list[i]['lam'] ** 2)) < 0.025

        # mf_analysis_full(X, scaling_ranges, gamint=gamint, p_exp=2,
        #                  n_cumul=3, q=q)
        # assert abs(lwt.cumulants.log_cumulants[1, :].mean()
        #            + (config_list[i]['lam'] ** 2)) < 0.025


@pytest.mark.bootstrap
def test_autorange(mrw_file):

    with open('tests/mrw_config.json', 'rb') as f:
        config_list = json.load(f)

    for i, fname in enumerate(mrw_file):

        with open(fname, 'rb') as f:
            X = np.load(f)

        if config_list[i]['H'] == 0.01:
            continue

        # j2 = int(np.log2(X.shape[0]) - 3)
        # wt_coefs, _, j2_eff, _ = wavelet_analysis(X, p_exp=None, j2=j2)

        # hmin = estimate_hmin(wt_coefs, j1=1, j2_eff=j2_eff, weighted=True)[0]
        # hmin = hmin.min()
        # gamint = 0.0 if hmin >= 0 else -hmin + 0.1

        j2 = int(np.log2(X.shape[0]) - 3)
        WT = wavelet_analysis(X, p_exp=2, j1=2, j2=12, weighted=None)

        coef_boot = WT.wt_coefs.bootstrap(5, 'db3')
        leader_boot = WT.wt_leaders.bootstrap(5, 'db3')

        gamint = 1
        j2_eff = coef_boot.j2_eff()

        q = np.array([2])
        scaling_ranges = [(3, j2) for j2 in range(3, j2_eff + 1)] + [(4, 1)]

        dwt, lwt = minimal_mf_analysis(
            WT.wt_coefs, WT.wt_leaders, j1=2, weighted='Nj', q=None,
            n_cumul=3)

        dwt_b, lwt_b = minimal_mf_analysis(
            coef_boot, leader_boot, j1=2, weighted='Nj', q=None, n_cumul=3)

        lwt_b.cumulants.compute_R()
        dwt_b.structure.compute_R()

        # minimal_mf_analysis(WT, scaling_ranges, gamint=gamint,p_exp=2, n_cumul=3, q=q)
        # assert abs(dwt.structure.H.mean() - config_list[i]['H']) < 0.1
        # assert abs(lwt.cumulants.log_cumulants[1, :].mean()
        #            + (config_list[i]['lam'] ** 2)) < 0.025

        # mf_analysis_full(X, scaling_ranges, gamint=gamint, p_exp=2,
        #                  n_cumul=3, q=q)
        # assert abs(lwt.cumulants.log_cumulants[1, :].mean()
        #            + (config_list[i]['lam'] ** 2)) < 0.025
