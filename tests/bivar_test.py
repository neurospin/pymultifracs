import pytest

import numpy as np
from scipy.io import loadmat

from pymultifracs.bivariate import bimfa


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

        # X = data[key].copy()
        # j1 = param[key]['j1'].squeeze()
        # j2 = param[key]['j2'].squeeze()
        # # j2 = int(np.log2(X.shape[0]) - 4)
        # p_exp = 2
        # gamint = 1

        # dwt, lwt = bivariate_analysis_full(X[:, 0], X[:, 1], p_exp=p_exp,
        #                                    scaling_ranges=[(j1, j2)],
        #                                    gamint=gamint, weighted=None,
        #                                    n_cumul=2,
        #                                    q1=np.array([0, 1, 2]),
        #                                    q2=np.array([0, 1, 2]),
        #                                    R=2)

        X = data[key].copy()

        # j1 = param[key]['j1'].squeeze() - 2
        j1 = 3
        j2 = int(np.log2(X.shape[0]) - 4)
        j2 = 9
        # j2 = param[key]['j2'].squeeze()
        p_exp = 2
        gamint = 1

        scaling_ranges = [(j1, j2)]

        dwt, lwt = bimfa(
            X[:, 0], X[:, 1], scaling_ranges, p_exp=p_exp, gamint=gamint,
            weighted=None, n_cumul=2, q1=np.array([0, 1, 2]),
            q2=np.array([0, 1, 2]), R=1)

        p = param[key]

        print(p['mft'][0, 0], lwt.cumulants.c20[0])
        print(p['mft'][0, 1], lwt.cumulants.c02[0])

        assert abs(p['mft'][0, 0] - np.sqrt(-lwt.cumulants.c20[0])) < 0.02
        assert abs(p['mft'][0, 1] - np.sqrt(-lwt.cumulants.c02[0])) < 0.02
        assert abs(p['mft'][0, 2] - lwt.cumulants.rho_mf[0]) < 0.11

    for key in data:
        test_key(key)
