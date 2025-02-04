import pytest

import numpy as np

from pymultifracs.estimation import estimate_hmin
from pymultifracs import mfa, wavelet_analysis
from pymultifracs.robust.benchmark import Benchmark, get_grid, get_fname
from pymultifracs.simul import mrw


@pytest.mark.benchmark
def test_benchmark_unit():

    signal_param_grid = {
        'H': np.array([.8]),
        'lam': np.array([np.sqrt(.05)]),
        'shape': np.array([2**16], dtype=int),
        # 'n_rep': [1],
    }

    def basic_mfa(WT, robust):

        WT = WT.integrate(1).get_leaders(np.inf)

        lwt = mfa(
            WT, scaling_ranges=[(3, 7)], robust=robust, weighted=None, n_cumul=2)

        return lwt.cumulants

    estimation_grid = {
        'Leader': lambda x: basic_mfa(x, False),
    }

    WT_params = {
        'wt_name': 'db6'
    }

    def load_generate_signals(param_grid):

        for signal_params in param_grid.itertuples(index=False):

            signal_params = signal_params._asdict()

            yield mrw(**signal_params), signal_params
            # fname = folder / get_fname(signal_params)

            # if fname.exists():
            #     yield np.load(fname), signal_params
            # else:
            #     raise ValueError('signal missing')

    bench = Benchmark(
        signal_param_grid, load_generate_signals, estimation_grid, WT_params)

    bench.compute_benchmark(n_jobs=1)
