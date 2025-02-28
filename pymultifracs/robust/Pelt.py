"""
Pelt implementation based on:
https://github.com/deepcharles/ruptures/blob/master/src/ruptures/detection/pelt.py
Which has been accelerated

Authors:
    Original authors of the Pelt python implementation.
    Merlin Dumeur <merlin@dumeur.net>.
"""

from math import floor

from joblib import Parallel, delayed
import numpy as np
from numba import jit

import ruptures as rpt
from ruptures.costs import cost_factory
from ruptures.base import BaseCost, BaseEstimator
from ruptures.exceptions import BadSegmentationParameters
from ruptures.utils import sanity_check

from .hilbert import _hilbert_cost


class Pelt(BaseEstimator):
    """Penalized change point detection.

    For a given model and penalty level, computes the segmentation which
    minimizes the constrained sum of approximation errors.
    """

    def __init__(self, model="l2", custom_cost=None, min_size=2, max_size=None,
                 jump=5, params=None, n_jobs=1):
        """Initialize a Pelt instance.

        Parameters
        ----------
        model : str, optional
            Segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'``
            is not None.
        custom_cost : rpt.BaseCost | None
            Custom cost function. Defaults to None.
        min_size : int, optional
            Minimum segment length.
        max_size : int | float, optional
            Maximum segment length, as a size or a fraction of the entire
            signal length.
        jump : int, optional
            Subsample (one every *jump* points).
        params : dict, optional
            A dictionary of parameters for the cost instance.
        """

        if custom_cost is not None:
            if isinstance(custom_cost, BaseCost):
                self.cost = custom_cost
            else:
                raise ValueError('custom_cost should inherit rpt.BaseCost')
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.max_size = max_size
        self.jump = jump
        self.n_samples = None
        self.n_jobs = n_jobs

    def _seg(self, pen):
        """Computes the segmentation for a given penalty using PELT (or a list
        of penalties).

        Args:
            penalty (float): penalty value

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """
        # initialization

        # partitions[t] contains the optimal partition of signal[0:t]
        partitions = dict()  # this dict will be recursively filled
        partitions[0] = {(0, 0): 0}
        admissible = []

        beta_factor = 1.1

        signal = self.cost.signal
        cost_w = self.cost.w

        # Recursion

        ind = [k for k in range(0, self.n_samples, self.jump)
               if k >= self.min_size]
        ind += [self.n_samples]

        # ind = np.arange((self.min_size // self.jump + 1) * self.jump,
        #                 self.n_samples, self.jump)
        # ind = np.r_[ind, self.n_samples]

        @jit(nopython=True)
        def get_subproblems(t, bkp, segment):
            # we update with the right partition
            # tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
            # tmp_partition = partitions[t].copy()
            # tmp_partition[(x, bkp)] = self.cost.error(x, bkp)  # + pen
            return _hilbert_cost(segment, cost_w)

        from .hilbert import _numba_mean, w_hilbert

        @jit(nopython=True)
        def _hilbert_cost2(X, w):

            mu = np.exp(_numba_mean(np.log(X), axis=0))

            for i in range(X.shape[0]):
                X[i] = w_hilbert(X[i], mu, w)

            return np.sum(X ** 2)

        cost_fun = _hilbert_cost2

        with Parallel(n_jobs=self.n_jobs, backend="threading", batch_size=1,
                      pre_dispatch='1*n_jobs') as parallel:

            for bkp in ind:
                # adding a point to the admissible set from the previous loop.
                # new_adm_pt = (bkp - self.min_size) // self.jump
                # new_adm_pt *= self.jump

                n_admissible_last = len(admissible)

                new_adm_pt = bkp - self.min_size
                new_adm_pt -= new_adm_pt % self.jump
                admissible.append(new_adm_pt)

                # Filter admissible partitions
                admissible = [
                    t for t in admissible if t in partitions
                ]

                subproblems = [
                    partitions[t].copy() for t in admissible
                ]

                for t, tmp_partition in zip(admissible, subproblems):
                    tmp_partition[(t, bkp)] = _hilbert_cost(
                        signal[t:bkp], cost_w)

                # compute = parallel(
                #     delayed(cost_fun)(
                #         signal[t:bkp].copy(), cost_w.copy())
                #     for t in admissible
                # )

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
                admissible = [
                    t for k, t in enumerate(admissible)
                    if costs[k] <= opt_cost + pen
                ]

                # if len(admissible) == n_admissible_last:
                # pen -= np.arctan(len(admissible) - 200) / np.pi * 2

                if len(admissible) > n_admissible_last:
                    if len(admissible) > 100:
                        pen *= .95
                else:
                    # pen += np.arctan(n_admissible_last - len(admissible)) / np.pi * 4
                    pen += .1 * (n_admissible_last - len(admissible))

                # trim segments that are too long

                if len(admissible) > 400:
                    admissible = [
                        t for t in admissible if bkp - t <= self.max_size
                    ]

                # print(bkp, pen)
                if bkp % 500 == 0:
                    print(pen, len(admissible))
                #     1/0

        best_partition = partitions[self.n_samples]
        best_partition.pop((0, 0))
        return best_partition

    def fit(self, signal) -> "Pelt":
        """Set params.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update params
        self.cost.fit(signal)
        if signal.ndim == 1:
            (n_samples,) = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples

        if type(self.max_size) is float:

            if self.max_size > 1:
                raise ValueError('If max_size is a float, it should be <= 1.0')

            self.max_size = floor(self.max_size * n_samples)

        if self.max_size is None:
            self.max_size = n_samples

        return self

    def predict(self, pen):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.pelt.Pelt.fit].

        Args:
            pen (float): penalty value (>0)

        Raises:
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        partition = self._seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, pen):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)
