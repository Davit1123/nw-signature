"""
Exact p-variation of discrete paths, accelerated with Numba.

For a discrete path x_0, ..., x_{T-1} in R^d and p >= 1 the functions compute

    ||x||_{p-var}^p = max over 0 = t_0 < t_1 < ... < t_k = T-1 of
                      sum_i |x_{t_(i+1)} - x_{t_i}|^p,

where |.| is the Euclidean norm. Note that they return the p-th power of the
p-variation; take the 1/p-th power for the p-variation itself.

The value is computed exactly by a dynamic programme over the points of the
path: the best partition ending at point j extends the best partition ending
at some earlier point m. Candidate points m are pruned using a dyadic table of
maximal increments, which avoids checking all O(T^2) pairs in practice.
"""

import math
from collections import namedtuple

import numba as nb
import numpy as np

PVar = namedtuple("PVar", ["value", "points"])


@nb.njit(cache=True, fastmath=True)
def _euclid(x, a, b):
    """Euclidean distance ||x[b] - x[a]|| for a path x of shape (T, d)."""
    s = 0.0
    for k in range(x.shape[1]):
        diff = x[b, k] - x[a, k]
        s += diff * diff
    return math.sqrt(s)


@nb.njit(cache=True, fastmath=True)
def _p_var_backbone(x, p):
    """
    p-th power of the p-variation of a path x of shape (T, d).

    Returns (value, point_links), where point_links[j] is the previous point
    of an optimal partition of x[0..j] that ends at j.
    """
    path_size = x.shape[0]
    if path_size == 0:
        return -math.inf, np.zeros(0, dtype=np.int64)
    if path_size == 1:
        return 0.0, np.zeros(1, dtype=np.int64)

    s = path_size - 1
    # N = floor(log2(s)) + 1
    N = 1
    while (s >> N) != 0:
        N += 1

    # ind[ind_n(j, n)]: maximal increment from the midpoint of a dyadic block to its points
    ind = np.zeros(s, dtype=np.float64)

    def ind_n(j, n):
        return (s >> n) + (j >> n)

    def ind_k(j, n):
        v = ((j >> n) << n) + (1 << (n - 1))
        return v if v < s else s

    max_p_var = 0.0
    run_p_var = np.zeros(path_size, dtype=np.float64)
    point_links = np.zeros(path_size, dtype=np.int64)

    for j in range(path_size):
        # update the dyadic table with point j
        for n in range(1, N + 1):
            if not ((j >> n) == (s >> n) and ((s >> (n - 1)) % 2) == 0):
                ii = ind_n(j, n)
                d = _euclid(x, ind_k(j, n), j)
                if d > ind[ii]:
                    ind[ii] = d

        if j == 0:
            continue

        # search the best predecessor m of j, skipping dyadic blocks that cannot improve
        m = j - 1
        delta = 0.0
        delta_m = j
        n = 0

        while True:
            while n > 0 and (m >> n) == (s >> n) and ((s >> (n - 1)) % 2) == 0:
                n -= 1

            skip = False
            if n > 0:
                iid = ind[ind_n(m, n)] + _euclid(x, ind_k(m, n), j)
                if delta >= iid:
                    skip = True
                elif m < delta_m:
                    delta = (max_p_var - run_p_var[m]) ** (1.0 / p)
                    delta_m = m
                    if delta >= iid:
                        skip = True

            if skip:
                k = (m >> n) << n
                if k > 0:
                    m = k - 1
                    while n < N and ((k >> n) % 2) == 0:
                        n += 1
                else:
                    break
            else:
                if n > 1:
                    n -= 1
                else:
                    d = _euclid(x, m, j)
                    if d >= delta:
                        new_p_var = run_p_var[m] + d ** p
                        if new_p_var >= max_p_var:
                            max_p_var = new_p_var
                            point_links[j] = m
                    if m > 0:
                        while n < N and ((m >> n) % 2) == 0:
                            n += 1
                        m -= 1
                    else:
                        break

        run_p_var[j] = max_p_var

    return run_p_var[path_size - 1], point_links


def p_variation(x, p=2.0):
    """
    p-variation of a single discrete path.

    Parameters
    ----------
    x : array-like of shape (n_timesteps, n_features) or (n_timesteps,)
        The path.
    p : float, default=2.0
        Exponent, p >= 1.

    Returns
    -------
    PVar(value, points)
        value is ||x||_{p-var}^p (the p-th power of the p-variation) and
        points are the indices of an optimal partition, starting at 0 and
        ending at n_timesteps - 1.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    value, point_links = _p_var_backbone(x, float(p))
    if x.shape[0] == 0:
        return PVar(value=value, points=[])

    points = [x.shape[0] - 1]
    while points[-1] != 0:
        points.append(int(point_links[points[-1]]))
    return PVar(value=value, points=points[::-1])


@nb.njit(parallel=True, cache=True, fastmath=True)
def pvar_backbone_matrix_numba(X1, X2, p):
    """
    Pairwise p-variation of difference paths.

    Parameters
    ----------
    X1 : ndarray of shape (n1, n_timesteps, n_features)
    X2 : ndarray of shape (n2, n_timesteps, n_features)
    p : float, p >= 1

    Returns
    -------
    M : ndarray of shape (n1, n2)
        M[i, j] = ||X1[i] - X2[j]||_{p-var}^p.
    """
    n1, T, d = X1.shape
    n2 = X2.shape[0]
    out = np.empty((n1, n2), dtype=np.float64)

    # flat parallel loop over all pairs
    for idx in nb.prange(n1 * n2):
        i = idx // n2
        j = idx - i * n2

        diff = np.empty((T, d), dtype=np.float64)
        for t in range(T):
            for k in range(d):
                diff[t, k] = X1[i, t, k] - X2[j, t, k]

        out[i, j] = _p_var_backbone(diff, p)[0]
    return out
