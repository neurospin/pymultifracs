import pytest
import json

import numpy as np
import matplotlib.pyplot as plt

from pymultifracs.estimation import estimate_hmin
from pymultifracs import mfa, wavelet_analysis
from pymultifracs.viz import plot_psd, plot_cm
from pymultifracs.utils import build_q_log

@pytest.mark.viz
def test_plots(fbm_file):

    # for i, fname in enumerate(fbm_file):

    with open(fbm_file[0], 'rb') as f:
        X = np.load(f)

    # j2 = int(np.log2(config_list[i]['shape']) - 3)

    plot_psd(X[:, 0], 1)

    WT = wavelet_analysis(X)

    WTpL = WT.get_leaders(2)

    WT.plot(3, 7)
    WTpL.plot(3, 7)

    pwt = mfa(WTpL, [(3, 7)], n_cumul=4, q=build_q_log(.1, 5, 5))

    pwt.structure.plot()
    pwt.structure.plot_scaling()
    pwt.cumulants.plot()
    fig, ax = plt.subplots()
    plot_cm(pwt.cumulants, 0, 3, 7, 0, ax)
    pwt.spectrum.plot()

        # scaling_ranges = [(3, WTL.j2_eff())]

        # WTL = WTL.auto_integrate(scaling_ranges)

        # assert WTL.gamint == 0

        # q = np.array([-2, -1, 0, 1, 2])

        # dwt, lwt = mfa([WTL.origin_mrq, WTL], scaling_ranges, n_cumul=4, q=q)

        # if config_list[i]['H'] != 0.01:
            # assert abs(dwt.structure.H.mean() - WTL.gamint - config_list[i]['H']) < 0.11
        # assert abs(lwt.cumulants.log_cumulants[1, :].mean()) < 0.0105

    return


@pytest.mark.bootstrap
def test_bootstrap(mrw_file):

    with open(mrw_file[0], 'rb') as f:
        X = np.load(f)

    WT = wavelet_analysis(X[:, :2])
    WTpL = WT.get_leaders(2)

    j2 = WTpL.max_scale_bootstrap()
    scaling_ranges = [(2, j2), (3, j2)]

    WT = WT.auto_integrate(scaling_ranges)

    dwt, lwt = mfa(
        [WT, WTpL], scaling_ranges, weighted='bootstrap', n_cumul=2,
        R=20, estimates='scm')

    dwt.structure.plot()

    lwt.cumulants.plot()
    lwt.spectrum.plot()
