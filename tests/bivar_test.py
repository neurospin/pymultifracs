import pytest

import numpy as np
from scipy.io import loadmat

from pymultifracs.bivariate import bimfa
from pymultifracs import wavelet_analysis


@pytest.mark.bivariate
def test_bivariate():

    names = ['nossnoMF', 'nossMF', 'ssMF', 'ssnoMF']

    data = {
        key: loadmat(f'tests/data/DataSet_{key}.mat')['data'].transpose()
        for key in names
    }

    param = {
        key: loadmat(f'tests/data/DataSet_{key}.mat')['params']
        for key in names
    }

    param = {
        key: {
            param[key].dtype.names[i]: param[key][0, 0][i]
            for i in range(len(param[key][0, 0]))
        }
        for key in param
    }

    def test_key(key):

        X = data[key].copy()

        j1 = 3
        j2 = int(np.log2(X.shape[0]) - 4)
        j2 = 9
        p_exp = 2
        gamint = .75

        scaling_ranges = [(j1, j2)]

        WT = wavelet_analysis(X)
        WTpL = WT.get_leaders(p_exp, gamint=gamint)

        lwt = bimfa(
            WTpL, WTpL, scaling_ranges, weighted=None, n_cumul=2,
            q1=np.array([0, 1, 2]), q2=np.array([0, 1, 2]), R=1)

        lwt.structure.get_jrange()
        lwt.structure.plot()
        lwt.cumulants.plot()
        lwt.cumulants.plot_legendre()

        p = param[key]

        assert abs(
            p['mft'][0, 0] - np.sqrt(-lwt.cumulants.c20[0, 0, 1])) < 0.02, key
        assert abs(
            p['mft'][0, 1] - np.sqrt(-lwt.cumulants.c02[0, 0, 1])) < 0.02
        assert abs(p['mft'][0, 2] - lwt.cumulants.rho_mf[0, 0, 1]) < 0.11

    for key in data:
        test_key(key)
