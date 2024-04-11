import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .. import wavelet_analysis, mfa
from .robust import get_outliers
from ..simul.noisy import gen_noisy


def estimate(gen_func, robust_cm=False, bootstrap_weight=False,
             outlier_detect=False, alpha=1, generalized=False,
             gen_func_kwargs=None, robust_kwargs=None):
    
    if gen_func_kwargs is None:
        gen_func_kwargs = {}

    if robust_kwargs is None:
        robust_kwargs = {}

    p_exp=2
    noisy_X = gen_func(**gen_func_kwargs)

    coefs = wavelet_analysis(
        noisy_X.reshape(noisy_X.shape[0], -1), wt_name='db6')
    
    coefs = coefs.integrate(gamint=1)

    R = 1
    weighted = None

    if bootstrap_weight:
        R = 25
        weighted = 'bootstrap'

    if outlier_detect:

        # try:

            _, idx_reject = get_outliers(
                coefs, [(6, 11)], robust_cm=robust_cm,
                generalized=generalized,
                hilbert_weighted=False, **robust_kwargs)

            lwt = mfa(coefs.get_leaders(p_exp), scaling_ranges=[(6, 11)],
                n_cumul=4, idx_reject=idx_reject, R=R, weighted=None)

        # except ValueError:
        #     return None
    
    else:

        try:
            lwt = mfa(
                coefs.get_leaders(p_exp), scaling_ranges=[(6, 11)],
                robust=robust_cm, R=R, weighted=weighted, n_cumul=4,
                robust_kwargs=robust_kwargs)
        except ValueError:
            return None

    return lwt


def gen_estimate(N, gen_func, SNRgrid, covgrid, align_scale, n_jobs=10,
                 n_rep=1, robust_cm=False, bootstrap_weight=False,
                 outlier_detect=False, lambd=None, generalized=False,
                 robust_kwargs=None):
    
    df_list = []

    for rep in tqdm(range(n_rep)):

        signal, noise = gen_func(N=N, lambd=lambd)

        SNRlist = np.array_split(SNRgrid, n_jobs)
        gen_func_kwargs = dict(
            signal=signal, noise=noise, coverage=covgrid,
            align_scale=align_scale, 
        )

        cms = Parallel(n_jobs=n_jobs)(
            delayed(estimate)(
                gen_noisy,
                robust_cm=robust_cm, bootstrap_weight=bootstrap_weight,
                outlier_detect=outlier_detect, alpha=1.1,
                generalized=generalized,
                gen_func_kwargs={**gen_func_kwargs, 'SNR': snr},
                robust_kwargs=robust_kwargs)
            for snr in SNRlist)
        
        # if outlier_detect:

        #     c1_dfs = [
        #         pd.DataFrame(cms[i][0].c1.squeeze().reshape(-1, covgrid.shape[0]),
        #                     index=SNR, columns=covgrid)
        #         for i, SNR in enumerate(SNRlist) if cms[i] is not None]
        #     c2_dfs = [
        #         pd.DataFrame(cms[i][0].c2.squeeze().reshape(-1, covgrid.shape[0]),
        #                     index=SNR, columns=covgrid)
        #         for i, SNR in enumerate(SNRlist) if cms[i] is not None]
        #     # c3_dfs = [
        #     #     pd.DataFrame(cms[i][0].c3.squeeze().reshape(-1, covgrid.shape[0]),
        #     #                 index=SNR, columns=covgrid)
        #     #     for i, SNR in enumerate(SNRlist)]

        c1_dfs = [
            pd.DataFrame(cms[i].cumulants.c1.squeeze().reshape(-1, covgrid.shape[0]),
                        index=SNR, columns=covgrid)
            for i, SNR in enumerate(SNRlist) if cms[i] is not None]
        c2_dfs = [
            pd.DataFrame(cms[i].cumulants.c2.squeeze().reshape(-1, covgrid.shape[0]),
                        index=SNR, columns=covgrid)
            for i, SNR in enumerate(SNRlist) if cms[i] is not None]
        # c3_dfs = [
        #     pd.DataFrame(cms[i].c3.squeeze().reshape(-1, covgrid.shape[0]),
        #                 index=SNR, columns=covgrid)
        #     for i, SNR in enumerate(SNRlist)]

        c1_df = pd.concat(c1_dfs).rename_axis(
            columns='coverage', index=['SNR'])
        c2_df = pd.concat(c2_dfs).rename_axis(
            columns='coverage', index=['SNR'])
        # c3_df = pd.concat(c3_dfs).rename_axis(
        #     columns='coverage', index=['SNR'])

        df = pd.concat([c1_df, c2_df], keys=['c1', 'c2'], axis=1,
                        names=['cumulant', 'coverage'])

        df_list.append(df)

    return pd.concat(df_list, axis=0, keys=np.arange(n_rep),
                     names=['rep', 'SNR'])
