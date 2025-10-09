"""
Implementation of the PELT algorithm, based on the ruptures package:
https://github.com/deepcharles/ruptures/blob/master/src/ruptures/detection/pelt.py

BSD 2-Clause License

Copyright (c) 2017-2022, ENS Paris-Saclay, CNRS
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors:
    Original authors of the Pelt python implementation.
    Merlin Dumeur <merlin@dumeur.net> (modifications of the PELT algorithm)
"""

from math import floor

import numpy as np
from numba import jit

from joblib import Parallel, delayed

from ruptures.costs import cost_factory
from ruptures.base import BaseCost


def pelt_segment(signal, pen, cost_fun, min_size=2,
                 max_size=None, jump=5, params=None, n_jobs=1,
                 trim_admissible=False, update_beta=False, verbose=False):

    if signal.ndim == 1:
        (n_samples,) = signal.shape
    else:
        n_samples, _ = signal.shape

    if type(max_size) is float:

        if max_size > 1:
            raise ValueError('If max_size is a float, it should be <= 1.0')

        max_size = floor(max_size * n_samples)

    if max_size is None:
        max_size = n_samples

    partitions = dict()  # this dict will be recursively filled
    partitions[0] = {(0, 0): 0}
    admissible = []

    # beta_factor = 1.1
    # if hasattr(cost, 'w'):
    #     cost_w = cost.w

    # Recursion

    ind = [k for k in range(0, n_samples, jump) if k >= min_size]
    ind += [n_samples]

    # ind = np.arange((self.min_size // self.jump + 1) * self.jump,
    #                 self.n_samples, self.jump)
    # ind = np.r_[ind, self.n_samples]

    # @jit(nopython=True)
    # def get_subproblems(t, bkp, segment):
    #     # we update with the right partition
    #     # tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
    #     # tmp_partition = partitions[t].copy()
    #     # tmp_partition[(x, bkp)] = self.cost.error(x, bkp)  # + pen
    #     return _hilbert_cost(segment, cost_w)

    # from .hilbert import _numba_mean, w_hilbert

    # @jit(nopython=True)
    # def _hilbert_cost2(X, w):

    #     mu = np.exp(_numba_mean(np.log(X), axis=0))

    #     for i in range(X.shape[0]):
    #         X[i] = w_hilbert(X[i], mu, w)

    #     return np.sum(X ** 2)

    # cost_fun = _hilbert_cost2

    @jit(nopython=False)
    def cost_wrapper(t, signal_view, i):
        return t, cost_fun(signal_view), i

    with Parallel(n_jobs=n_jobs, backend="threading", batch_size=1,
                  pre_dispatch='1*n_jobs') as parallel:

        for bkp in ind:
            # adding a point to the admissible set from the previous loop.
            # new_adm_pt = (bkp - self.min_size) // self.jump
            # new_adm_pt *= self.jump

            n_admissible_last = len(admissible)

            new_adm_pt = bkp - min_size
            new_adm_pt -= new_adm_pt % jump  # TODO: check corectness of this?
            admissible.append(new_adm_pt)

            # Filter admissible partitions
            admissible = [t for t in admissible if t in partitions]

            subproblems = [partitions[t].copy() for t in admissible]

            # for t, tmp_partition in zip(admissible, subproblems):
            #     tmp_partition[(t, bkp)] = cost.error(signal[t:bkp])

            compute = parallel(
                delayed(cost_wrapper)(t, signal[t:bkp], i)
                for i, t in enumerate(admissible)
            )

            for t, c, i in compute:
                subproblems[i][(t, bkp)] = c

            # for k, (t, cost) in enumerate(zip(admissible, compute)):
            #     subproblems[k][(t, bkp)] = cost

            # partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
            # admissible = [
            #     t
            #     for t, partition in zip(admissible, subproblems)
            #     if sum(partition.values()) <= sum(partitions[bkp].values()) + pen
            # ]

            # finding the optimal partition
            costs = [sum(d.values()) + pen * len(d.values())
                     for d in subproblems]
            opt_part = np.argmin(costs)
            partitions[bkp] = subproblems[opt_part]
            opt_cost = costs[opt_part]

            # trimming the admissible set
            admissible = [t for k, t in enumerate(admissible)
                          if costs[k] <= opt_cost + pen]

            # if len(admissible) == n_admissible_last:
            # pen -= np.arctan(len(admissible) - 200) / np.pi * 2

            # dynamically upate beta
            if update_beta:
                if len(admissible) > n_admissible_last:
                    if len(admissible) > 100:
                        pen *= .95
                else:
                    # pen += np.arctan(n_admissible_last - len(admissible)) / np.pi * 4
                    pen += .1 * (n_admissible_last - len(admissible))

            # if self.trim_admissible and len(admissible) > 400:
            # trim segments that are longer than max_size
            if len(admissible) > 400:
                admissible = [
                    t for t in admissible if bkp - t <= max_size
                ]

            # print(bkp, pen)
            if verbose and bkp % 500 == 0:
                print(pen, len(admissible))

    best_partition = partitions[n_samples]
    best_partition.pop((0, 0))
    return best_partition
