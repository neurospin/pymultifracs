"""
Authors: Merlin Dumeur <merlin@dumeur.net>
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .. import wavelet_analysis, mfa
from .robust import get_outliers
from ..simul.noisy import gen_noisy


def get_grid(param_grid):
    """
    Creates a grid from a parameter dictionary.
    """

    series = [
        pd.DataFrame({name: signal_param})
        for name, signal_param in param_grid.items()]

    out = series[0]

    for s in series[1:]:
        out = out.merge(s, how='cross')

    return out


def get_fname(param):
    """
    Create filename from parameters.
    """

    fname = Path('.')

    for param in param.values():

        if isinstance(param, float):
            fname /= f'{param:.2f}'
        else:
            fname /= str(param)

    fname /= 'signal'

    return fname.with_suffix('.npy')


@dataclass
class Benchmark:
    """
    Compute and plot benchmarks.
    """
    signal_param_grid: dict[str, np.ndarray]
    # noise_param_grid: dict[str, np.ndarray]
    signal_func: Callable
    # noise_gen_func: Callable
    estimation_grid: dict[str, Callable]
    WT_params: dict[str, Any]
    # parameters_df: pd.DataFrame = field(init=False, default=None, repr=False)
    results: pd.DataFrame = field(init=False, repr=False)

    def get_df_fnames(self):
        """
        Get results filename.
        """
        return Path('results.pkl')

    def generate_grids(self):
        """
        Generate parameter grid.
        """

        signal_grid = get_grid(self.signal_param_grid)
        # noise_grid = get_grid(self.noise_param_grid)

        return signal_grid  # , noise_grid

    def load_df(self):
        """
        Load results dataframe.
        """

        results_fname = self.get_df_fnames()
        # self.parameters_df = self.generate_grids()

        if results_fname.exists():
            self.results = pd.read_pickle(results_fname)

    def compute_benchmark(self, n_jobs=1, save=False):
        """
        Compute the benchmark.
        """

        results = {}

        signal_grid = get_grid(self.signal_param_grid)
        signal_names = signal_grid.columns
        # print(signal_names)
        # signals, signal_names = self.load_generate_signals()

        def estimate_mf(signal, signal_params):
            res = []
            WT = wavelet_analysis(signal, **self.WT_params)
            for method, est_fun in self.estimation_grid.items():
                res.append((method, est_fun(WT)))

            return res, signal_params

        print(signal_grid)
        results = Parallel(n_jobs=n_jobs)(
            delayed(estimate_mf)(*s)
            for s in tqdm(self.signal_func(signal_grid),
                          total=signal_grid.shape[0]))

        results = {
            (method, *signal_params.values()): [estimate]
            for res_list, signal_params in results
            for method, estimate in res_list
        }

        self.results = pd.DataFrame.from_dict(results).transpose()

        self.results.index.names = ['method', *signal_names]
        self.results.columns = ['cumulants']

        results_fname = self.get_df_fnames()
        results_fname.parent.mkdir(parents=True, exist_ok=True)

        self.results.to_pickle(results_fname)

    def plot(self):
        """
        Plotting function. Empty for now.
        """
        return


def estimate(gen_func, robust_cm=False, bootstrap_weight=False,
             outlier_detect=False, alpha=1, generalized=False,
             gen_func_kwargs=None, robust_kwargs=None):
    """
    Automated estimation.
    """

    if gen_func_kwargs is None:
        gen_func_kwargs = {}

    if robust_kwargs is None:
        robust_kwargs = {}

    p_exp = 2
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

        _, idx_reject = get_outliers(
            coefs, [(6, 11)], robust_cm=robust_cm,
            generalized=generalized,
            hilbert_weighted=False, **robust_kwargs)

        lwt = mfa(coefs.get_leaders(p_exp), scaling_ranges=[(6, 11)],
                  n_cumul=4, idx_reject=idx_reject, R=R, weighted=None)

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
    """
    Generate signals and estimate multifractal properties.
    """

    df_list = []

    for _ in range(n_rep):

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

        c1_dfs = [
            pd.DataFrame(
                cms[i].cumulants.c1.squeeze().reshape(-1, covgrid.shape[0]),
                index=SNR, columns=covgrid)
            for i, SNR in enumerate(SNRlist) if cms[i] is not None]
        c2_dfs = [
            pd.DataFrame(
                cms[i].cumulants.c2.squeeze().reshape(-1, covgrid.shape[0]),
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
