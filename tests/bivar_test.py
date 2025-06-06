import pytest

import numpy as np
from scipy.io import loadmat

from pymultifracs.bivariate import bimfa
from pymultifracs import wavelet_analysis


@pytest.mark.bivariate
def test_bivariate_performance():

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

        p = param[key]

        assert abs(
            p['mft'][0, 0] - np.sqrt(-lwt.cumulants.c20.sel(channel1=0, channel2=1, scaling_range=0))) < 0.02, key
        assert abs(
            p['mft'][0, 1] - np.sqrt(-lwt.cumulants.c02[0, 0, 1])) < 0.02
        assert abs(p['mft'][0, 2] - lwt.cumulants.rho_mf[0, 0, 1]) < 0.11

    for key in data:
        test_key(key)

@pytest.mark.bivariate
def test_bivariate_unit():

    names = ['ssMF']

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

        bimfa(
            WTpL, WTpL, scaling_ranges, weighted=None, n_cumul=2,
            q1=None, q2=None, R=1, min_j='auto')

        with pytest.raises(ValueError):
            bimfa(
                WTpL, WTpL, [], weighted=None, n_cumul=2,
                q1=None, q2=None, R=1, min_j='auto')

        with pytest.raises(ValueError):
            bimfa(
                WTpL, WTpL, [], weighted=None, n_cumul=2,
                q1=None, q2=None, R=1, min_j=j1+1
            )

    for key in data:
        test_key(key)


@pytest.mark.bivariate
def test_bivariate_bootstrap_integration():

    names = ['ssMF']

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
        R = 5

        scaling_ranges = [(j1, j2)]

        WT = wavelet_analysis(X)
        WTpL = WT.get_leaders(p_exp, gamint=gamint)

        bimfa(
            WTpL, WTpL, scaling_ranges, weighted=None, n_cumul=2,
            q1=None, q2=None, R=R, min_j='auto')

    for key in data:
        test_key(key)


@pytest.mark.bivariate
def test_bivariate_plots():

    names = ['nossnoMF', 'ssMF']

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

        lwt.structure.plot()
        lwt.cumulants.plot()
        lwt.cumulants.plot_legendre()

    for key in data:
        test_key(key)
