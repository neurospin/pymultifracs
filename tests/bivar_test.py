import pytest

import numpy as np
from scipy.io import loadmat

from pymultifracs.bivariate import bivariate_analysis_full


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

        X = data[key]
        j1 = 3
        j2 = np.log2(X.shape[0]) - 3
        p_exp=2
        gamint = 1

        dwt, lwt = bivariate_analysis_full(X[:, 0], X[:, 1], p_exp=p_exp,
                                           j1=j1, j2=j2, gamint=gamint,
                                           weighted=True, n_cumul=2,
                                           q1=np.array([0, 1, 2]),
                                           q2=np.array([0, 1, 2]))

        p = param[key]

        assert abs(p['mft'][0, 0] - np.sqrt(-lwt.cumulants.c20[0])) < 0.02
        assert abs(p['mft'][0, 1] - np.sqrt(-lwt.cumulants.c02[0])) < 0.02
        assert abs(p['mft'][0, 2] - lwt.cumulants.rho_mf[0]) < 0.1

    for key in data:
        test_key(key)
