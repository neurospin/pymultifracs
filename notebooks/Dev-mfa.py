import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pymultifracs import wavelet_analysis, mfa
    from pymultifracs.simul import mrw
    from pymultifracs.utils import build_q_log
    import numpy as np
    import matplotlib.pyplot as plt
    return build_q_log, mfa, mrw, np, plt, wavelet_analysis


@app.cell
def _(mrw, np):
    N = 2 ** 16
    lam = np.sqrt(.05)
    H = .8
    X = mrw(H=H, lam=lam, shape=N, L=N)
    return H, N, X, lam


@app.cell
def _(X, wavelet_analysis):
    WTpL = wavelet_analysis(X).get_leaders(2)
    return (WTpL,)


@app.cell
def _(WTpL, build_q_log, mfa):
    pwt = mfa(WTpL, [(3, 13)], q=build_q_log(.1, 4, 20))
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


@app.cell
def _(pwt):
    pwt.spectrum.hq
    return


@app.cell
def _(plt, pwt):
    pwt.spectrum.plot()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
