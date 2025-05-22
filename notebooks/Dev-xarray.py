import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    from pymultifracs import wavelet_analysis
    from pymultifracs.bivariate import bimfa
    from pymultifracs.simul import mrw
    import numpy as np
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    import xarray as xr
    import pooch
    return bimfa, loadmat, np, plt, pooch, wavelet_analysis


@app.cell
def _(loadmat, pooch):
    url = ('https://github.com/neurospin/pymultifracs/raw/refs/heads/master/tests/'
           'data/DataSet_ssMF.mat')
    fname = pooch.retrieve(url=url, known_hash=None, path=pooch.os_cache("bivar"))
    X = loadmat(fname)['data'].T
    return (X,)


@app.cell
def _(X, bimfa, np, wavelet_analysis):
    WT = wavelet_analysis(X)
    WTpL = WT.get_leaders(2, gamint=1)

    lwt = bimfa(
        WTpL, WTpL, [(3, 9)], weighted=None, n_cumul=2,
        q1=np.array([0, 1, 2]), q2=np.array([0, 1, 2]), R=1)
    return (lwt,)


@app.cell
def _(lwt, plt):
    lwt.cumulants.plot()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
