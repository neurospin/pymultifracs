import pytest
import json

import numpy as np

from pymultifracs import mfa, wavelet_analysis


@pytest.mark.bootstrap
def test_confidence_interval(mrw_file):

    with open('tests/mrw_config.json', 'rb') as f:
        config_list = json.load(f)

    for i, fname in enumerate(mrw_file):

        with open(fname, 'rb') as f:
            X = np.load(f)

        WT = wavelet_analysis(X[:, :20])
        WTpL = WT.get_leaders(2)

        j2 = WTpL.max_scale_bootstrap()
        scaling_ranges = [(2, j2), (3, j2)]

        WT = WT.auto_integrate(scaling_ranges)

        dwt, lwt = mfa(
            [WT, WTpL], scaling_ranges, weighted='bootstrap', n_cumul=2,
            R=5, estimates='sc')

        lwt.cumulants.compute_Lambda()
        lwt.cumulants.get_jrange(1, 2, True)

        dwt.structure.compute_Lambda()

        dwt.structure.CIE_S_q(2)
        dwt.structure.CI_S_q(2)
        dwt.structure.CI_s_q(2)
        dwt.structure.CIE_s_q(2)

        lwt.cumulants.CI_c2
        lwt.cumulants.CIE_c2
        lwt.cumulants.CI_C2
        lwt.cumulants.CIE_C2

        lwt.cumulants.VE_c2
        lwt.cumulants.SE_c2
        lwt.cumulants.V_c2
        lwt.cumulants.STD_c2

        assert abs(
            dwt.structure.H[0].mean() - WT.gamint - config_list[i]['H'] < 0.1)
        assert abs(lwt.cumulants.c2[0].mean()
                   + (config_list[i]['lam'] ** 2)) < 0.025
