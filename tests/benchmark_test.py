import pytest

import numpy as np

from pymultifracs.estimation import estimate_hmin
from pymultifracs import mfa, wavelet_analysis
from pymultifracs.robust.benchmark import Benchmark, get_grid, get_fname
from pymultifracs.simul import mrw


@pytest.mark.benchmark
def test_benchmark_unit():

    signal_param_grid = {
        'mrw': {
            'H': np.array([.8]),
            'lam': np.array([np.sqrt(.05)]),
            'shape': np.array([2**16], dtype=int),
        }
    }

    def mrw_gen(H, lam, shape):
        return mrw(H=H, shape=int(shape), L=int(shape), lam=lam)

    signal_gen_grid = {
        'mrw': mrw_gen
    }

    def basic_mfa(X, p_exp, robust=False):

        WT = wavelet_analysis(
            X, wt_name='db6').integrate(1).get_leaders(p_exp)

        lwt = mfa(
            WT, scaling_ranges=[(3, 7)], robust=robust, weighted=None, n_cumul=2)

        return lwt.cumulants.log_cumulants.to_dataframe('cm').unstack('m')

        # return {
        #     'c1': lwt.cumulants.c1.to_dataframe(name='c1'),
        #     'c2': lwt.cumulants.c2.to_dataframe(name='c2')
        # }

    estimation_param_grid = {
        'leader': {
            'p_exp': np.array([0.5, 1]),
        },
    }

    estimation_grid = {
        'leader': basic_mfa
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
        signal_gen_grid, signal_param_grid, estimation_grid,
        estimation_param_grid)

    bench.compute_benchmark(n_jobs=1)
