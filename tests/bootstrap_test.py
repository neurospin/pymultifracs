import pytest
import json

import numpy as np

from pymultifracs.wavelet import decomposition_level_bootstrap
from pymultifracs import mfa, wavelet_analysis


# @pytest.mark.bootstrap
# def test_wavelet_bootstrap(mrw_file):

#     for fname in mrw_file:

#         with open(fname, 'rb') as f:
#             X = np.load(f)

#         j2 = 8

#         WT = wavelet_analysis(X, j2=j2).get_leaders(p_exp=np.inf)
#         WT.bootstrap(5)


@pytest.mark.bootstrap
def test_confidence_interval(mrw_file):

    with open('tests/mrw_config.json', 'rb') as f:
        config_list = json.load(f)

    for i, fname in enumerate(mrw_file):

        with open(fname, 'rb') as f:
            X = np.load(f)

        j2 = decomposition_level_bootstrap(X, 'db3')
        scaling_ranges = [(2, j2)]

        WT = wavelet_analysis(X, j2=j2)
        WTpL = WT.get_leaders(p_exp=2)

        dwt, lwt = mfa(
            [WT, WTpL], scaling_ranges, weighted='bootstrap', n_cumul=2,
            R=5, estimates='sc')

        print(
            dwt.structure.S_q(2).shape,
            dwt.structure.bootstrapped_mrq.S_q(2).shape)

        dwt.structure.CIE_S_q(2)
        dwt.structure.CI_S_q(2)
        dwt.structure.CI_s_q(2)
        dwt.structure.CIE_s_q(2)

        lwt.cumulants.CI_c2
        lwt.cumulants.CIE_c2
        lwt.cumulants.CI_C2
        lwt.cumulants.CIE_C2

        assert abs(dwt.structure.H.mean() - config_list[i]['H'] < 0.1)
        assert abs(lwt.cumulants.log_cumulants[1, :].mean()
                   + (config_list[i]['lam'] ** 2)) < 0.025


@pytest.mark.bootstrap
def test_autorange(mrw_file):

    for i, fname in enumerate(mrw_file):

        with open(fname, 'rb') as f:
            X = np.load(f)

        j2 = decomposition_level_bootstrap(X, 'db3')
        scaling_ranges = [(2, j2), (3, j2)]

        WT = wavelet_analysis(X, j2=j2)
        WTpL = WT.get_leaders(p_exp=2)

        dwt, lwt = mfa(
            [WT, WTpL], scaling_ranges, weighted='bootstrap', n_cumul=2,
            R=5, estimates='sc')

        lwt.cumulants.compute_Lambda()
        dwt.structure.compute_Lambda()
