"""
Authors: Merlin Dumeur <merlin@dumeur.net>
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable
from functools import reduce

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .. import wavelet_analysis, mfa
from .robust import get_outliers
from ..simul.noisy import gen_noisy


def get_grid(param_grid, func_grid):
    """
    Creates a grid from a parameter and a function dictionary.
    """

    param_dfs = {
        fun: [pd.DataFrame({name: signal_param})
              for name, signal_param in param_grid[fun].items()]
        for fun in func_grid
    }

    merge = lambda left, right: pd.merge(left, right, how='cross')

    # replace empty parameter dataframes
    for fun in param_dfs:
        if len(param_dfs[fun]) == 0:
            param_dfs[fun] = [pd.DataFrame(index=[fun])]

    param_dfs = {
        fun: reduce(merge, param_dfs[fun])
        for fun in param_dfs
    }

    return pd.concat(param_dfs, names=['model', 'parameters'])


def get_fname(name, param, folder='.'):
    """
    Create filename from parameters.
    """

    fname = Path(folder)

    for param in [*dict(param).values()]:

        if isinstance(param, float):
            fname /= f'{param:.2f}'
        else:
            fname /= str(param)

    fname /= name

    return fname.with_suffix('.npy')


@dataclass
class Benchmark:
    """
    Compute and plot benchmarks, varying the models and analysis parameters.

    Attributes
    ----------
    signal_gen_grid : dict[str, Callable]
        Dictionary associating a name to a function that samples time series.
    signal_param_grid : dict[str, ndarray of Any] | \
            dict[str, dict[str, ndarray of Any]]
        Dictionary associating to each signal generating function the
        dictionary associating to each signal parameter the array of the
        values that the parameter will take. May be empty.
    estimation_grid : dict[str, Callable]
        Dictionary associating to an estimation method's name its callable
        function.
    estimation_param_grid : dict[str, dict[str, ndarray of Any]]
        Dictionary associating to each estimation method the dictionary of
        estimation parameters. May be empty.
    folder : str | Path
        Path to the folder which will contain the output files.
    results : DataFrame
        Dataframe collecting the outcomes of the estimation on the generated
        signals.

    """
    signal_gen_grid: dict[str, Callable]
    signal_param_grid: dict[str, dict[str, np.ndarray]]
    # noise_param_grid: dict[str, np.ndarray]
    # noise_gen_func: Callable
    estimation_grid: dict[str, Callable]
    estimation_param_grid: dict[str, dict[str, Callable]]
    # WT_params: dict[str, Any]
    folder: str | Path = '.'
    # parameters_df: pd.DataFrame = field(init=False, default=None, repr=False)
    results: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self.folder = Path(self.folder)
        self._load_df()

    def get_df_fnames(self):
        """
        Get results filename.

        Returns
        -------
        Path
            Path to the results filename (if saved)
        """
        return Path(self.folder / 'results.pkl')

    def _generate_grids(self):
        """
        Generate signal generation and estimation parameter grids.
        """

        signal_grid = get_grid(self.signal_param_grid, self.signal_gen_grid)

        estimation_grid = get_grid(
            self.estimation_param_grid, self.estimation_grid)
        # noise_grid = get_grid(self.noise_param_grid)

        return signal_grid, estimation_grid  # , noise_grid

    def _generate_load_signal(self, name, params, save_load=False):

        restricted_params = {
            k: v for k, v in params.items()
            if k in self.signal_param_grid[name]
        }

        fname = get_fname(name, params, self.folder)

        if save_load and fname.exists():
            return np.load(fname)

        out = self.signal_gen_grid[name](**dict(params))

        if save_load:
            np.save(fname, out)

        return out

    def _load_df(self):
        """
        Load results dataframe.

        Currently loads systematically the latest results dataframe
        (disregards changes in parameters)
        """

        results_fname = self.get_df_fnames()
        # self.parameters_df = self.generate_grids()

        if results_fname.exists():
            print('Latest results dataframe found and loaded')
            self.results = pd.read_pickle(results_fname)

    def compute_benchmark(self, n_jobs=1, save_load_signals=False, save=False):
        """
        Compute the benchmark.

        Parameters
        ----------
        n_jobs : int
            Number of jobs that joblib will start in parallel. Each job handles
            a different signal configuration.
        save_load_signals : bool
            Whether to save the signals, and load them if found, to speed up
            computation. Currently not implemented yet.
        save : bool
            Whether the save the final results dataframe to a pickled file.
        """

        results = {}

        signal_param_grid, estimation_param_grid = self._generate_grids()

        # print(estimation_param_grid)

        # N_signals = sum(df.shape[0] for df in signal_param_grid.values())

        # def signal_iterator():
        #     for fun in signal_param_grid:
        #         for _, param_set in signal_param_grid[fun].iterrows():
        #             yield fun, param_set

        # signals_it = signal_iterator()

        def estimate_mf(signal, model, signal_params):

            # out = None

            res = []

            for param_tuple in estimation_param_grid.itertuples():

                method, _ = param_tuple.Index

                est_params = param_tuple._asdict()
                est_params.pop('Index')

                est_fun = self.estimation_grid[method]

                # only feed the parameters expected by the estimation function
                param_restriction = {
                    k: v for k, v in est_params.items()
                    if k in self.estimation_param_grid[method]
                }

                df = pd.DataFrame.from_dict(
                    est_fun(signal, **param_restriction))
                df = df.assign(**est_params, method=method)
                df.index.name = 'k'

                res.append(df)

            res = pd.concat(res)

            res = res.assign(model=model, **signal_params)
            res = res.reset_index().set_index(
                ['model', *signal_params, 'method',
                 *estimation_param_grid.columns, 'k'])

            return res

        # if save_load_signals:
        #     signal_gen_wrap = lambda x, y: self._generate_load_signal(x, y, True)

        # else:
        # # Wrapping signal generation to not feed the wrong parameters
        # def signal_gen_wrap(model, signal_params):

        #     signal_params = signal_params.to_dict()

        #     restricted_params = {
        #         k: v for k, v in signal_params.items()
        #         if k in self.signal_param_grid[model]
        #     }

        #     return self.signal_gen_grid[model](**restricted_params)

        def signal_gen_wrap(name, signal_params):

            restricted_params = {
                k: v for k, v in signal_params.items()
                if k in self.signal_param_grid[name]
            }

            fname = get_fname(name, signal_params, self.folder)

            if save_load_signals and fname.exists():
                return np.load(fname)

            out = self.signal_gen_grid[name](**signal_params)

            if save_load_signals:
                np.save(fname, out)

            return out

        def iterate_tuples(grid):

            for tup in grid.itertuples():

                index, _ = tup.Index

                data = tup._asdict()
                data.pop('Index')

                yield index, data

        results = Parallel(n_jobs=n_jobs)(
            delayed(estimate_mf)(
                signal_gen_wrap(index, data), index, data)
            for index, data in tqdm(
                iterate_tuples(signal_param_grid),
                total=signal_param_grid.shape[0])
        )

        self.results = pd.concat(results).sort_index()

        if save:

            results_fname = self.get_df_fnames()
            results_fname.parent.mkdir(parents=True, exist_ok=True)

            self.results.to_pickle(results_fname)

    def _plot(self):
        """
        Plotting function. Empty for now.
        """
        return
