import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pymultifracs import wavelet_analysis, mfa
    from pymultifracs.simul import mrw
    import numpy as np
    import matplotlib.pyplot as plt
    return mfa, mrw, np, plt, wavelet_analysis


@app.cell
def _(mrw, np):
    N = 2 ** 16
    lam = np.sqrt(.05)
    H = .8
    X = mrw(H=H, lam=lam, shape=N, L=N)
    return (X,)


@app.cell
def _(X, wavelet_analysis):
    WTpL = wavelet_analysis(X).get_leaders(2)
    return (WTpL,)


@app.cell
def _(WTpL, mfa):
    pwt = mfa(WTpL, [(3, 13)])
    return (pwt,)


@app.cell
def _(WTpL):
    WTpL.values[13]
    return


@app.cell
def _(pwt):
    pwt.cumulants.values.sel(j=13)
    return


@app.cell
def _(plt, pwt):
    pwt.cumulants.plot()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
