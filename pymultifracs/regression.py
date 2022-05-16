import numpy as np


def prepare_weights(mrq, weighted, n_ranges, j_min, j_max, scaling_ranges,
                    std=None):

    if weighted == 'Nj':

        w = np.tile(mrq.get_nj_interv(j_min, j_max)[None, :, None, :],
                    (1, 1, n_ranges, 1))

    elif weighted == 'bootstrap':

        std[std == 0] = std[std != 0].min()

        # std shape (n_moments, n_scales) -> (n_moments, n_scales, n_scaling_ranges, n_rep)
        if len(std.shape) == 2:
            w = np.tile(1 / std[:, :, None, None], (1, 1, n_ranges, 1))
        # std shape (n_moments, n_scales, n_scaling_ranges) -> (n_moments, n_scales, n_scaling_ranges, n_rep)
        else:
            w = std[:, :, :, None]

    else:  # weighted is None
        w = np.ones((1, j_max - j_min + 1, n_ranges, 1))

    for i, (j1, j2) in enumerate(scaling_ranges):

        w[:, j2-j_min+1:, i, :] = 0
        w[:, :j1-j_min, i, :] = 0

    # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    return w


def prepare_regression(scaling_ranges, j):

    n_ranges = len(scaling_ranges)
    j_min = min([sr[0] for sr in scaling_ranges])
    j_max = max([sr[1] for sr in scaling_ranges])

    # j_min = j.min()
    # j_max = j.max()

    # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    x = np.arange(j_min, j_max + 1)[None, :, None, None]

    return x, n_ranges, j_min, j_max, j_min - j.min(), j_max - j.min() + 1


def linear_regression(x, y, nj, return_variance=False):
    """
    Performs a (weighted or not) linear regression.
    Finds 'a' that minimizes the error:
        sum_j { n[j]*(y[j] - (a*x[j] + b))**2 }

    Args:
        x, y : regression variables
        nj: list containg the weigths
    Returns:
        a, b: angular coefficient and intercept

    (!!!!!!!!!!!!!)
    IMPORTANT:

    return_variance NOT DEBUGGED
    (!!!!!!!!!!!!!)
    """

    # bj = np.array(nj, dtype=np.float)
    assert isinstance(nj, np.ndarray)
    assert nj.shape[1] == x.shape[1]

    # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    V_0 = np.nansum(nj, axis=1)[:, None, :, :]
    V_1 = np.nansum(nj * x, axis=1)[:, None, :, :]
    V_2 = np.nansum(nj * (x**2), axis=1)[:, None, :, :]

    weights_slope = nj * (V_0*x - V_1)/(V_0*V_2 - V_1*V_1)
    weights_intercept = nj * (V_2 - V_1*x)/(V_0*V_2 - V_1*V_1)

    a = np.nansum(weights_slope*y, axis=1)
    b = np.nansum(weights_intercept*y, axis=1)

    wt = np.zeros_like(nj)
    wt[nj != 0] = 1 / nj[nj != 0]
    var_a = np.nansum(wt*weights_slope*weights_slope, axis=1)

    if not return_variance:
        return a, b
    else:
        return a, b, var_a


def compute_R2(moment, slope, intercept, weights, j_min, j_max):

    weights = 1 / weights

    # Shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    moment = moment[:, j_min-1:j_max, None, :]
    slope = slope[:, None, :]
    intercept = intercept[:, None, :]

    x = np.arange(j_min, j_max + 1)[None, :, None, None]

    res = (weights ** 2 * (moment - x * slope - intercept) ** 2).sum(axis=1)

    avg = (moment * weights).mean(axis=1)[:, None, :]
    tot = ((moment * weights - avg) ** 2).sum(axis=1)

    return 1 - res / tot
