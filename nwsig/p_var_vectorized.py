import math
import numpy as np
import numba as nb
from collections import namedtuple

@nb.njit(cache=True, fastmath=True)
def _p_var_backbone_abs_numba(x, p):
    """
    Numba-accelerated version of your p_var_backbone when
    path_dist(a,b) = |x[b] - x[a]|.
    Returns (value, points_array).
    """
    path_size = x.shape[0]

    if path_size == 0:
        return -math.inf, np.empty(0, dtype=np.int64)
    if path_size == 1:
        return 0.0, np.array([0], dtype=np.int64)

    s = path_size - 1
    # N = floor(log2(s)) + 1
    N = 1
    while (s >> N) != 0:
        N += 1

    # ind has length s (same as original)
    ind = np.zeros(s, dtype=np.float64)

    def ind_n(j, n):
        return (s >> n) + (j >> n)

    def ind_k(j, n):
        v = ((j >> n) << n) + (1 << (n - 1))
        return v if v < s else s  # min(…, s)

    max_p_var = 0.0
    run_p_var = np.zeros(path_size, dtype=np.float64)
    point_links = np.zeros(path_size, dtype=np.int64)

    for j in range(path_size):
        # precompute 'ind' row for this j
        for n in range(1, N + 1):
            # same guard as your code
            if not ( (j >> n) == (s >> n) and ((s >> (n - 1)) % 2) == 0 ):
                ii = ind_n(j, n)
                kk = ind_k(j, n)
                d = abs(x[kk] - x[j])
                if d > ind[ii]:
                    ind[ii] = d

        if j == 0:
            continue

        m = j - 1
        delta = 0.0
        delta_m = j
        n = 0

        while True:
            while n > 0 and (m >> n) == (s >> n) and ((s >> (n - 1)) % 2) == 0:
                n -= 1

            skip = False
            if n > 0:
                iid = ind[ind_n(m, n)] + abs(x[ind_k(m, n)] - x[j])
                if delta >= iid:
                    skip = True
                elif m < delta_m:
                    # NOTE: same as your pow(max_p_var - run_p_var[m], 1/p)
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
                    d = abs(x[m] - x[j])
                    if d >= delta:
                        new_p_var = run_p_var[m] + (d ** p)
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

    # reconstruct points (reverse of the chain from s back to 0)
    pts = np.empty(path_size, dtype=np.int64)
    cnt = 0
    point_i = s
    while True:
        pts[cnt] = point_i
        cnt += 1
        if point_i == 0:
            break
        point_i = point_links[point_i]

    # reverse first cnt entries
    for t in range(cnt // 2):
        a = t
        b = cnt - 1 - t
        pts[a], pts[b] = pts[b], pts[a]

    return run_p_var[path_size - 1], pts[:cnt]


# ---------- Friendly Python wrappers ----------

_PVar = namedtuple('p_var', ['value', 'points'])

def p_var_backbone_abs_fast(x: np.ndarray, p: float = 2.0) -> _PVar:
    """
    Drop-in replacement for p_var_backbone when path_dist(a,b)=|x[b]-x[a]|.
    x: shape (T,)
    """
    x = np.asarray(x, dtype=np.float64)
    val, pts = _p_var_backbone_abs_numba(x, float(p))
    return _PVar(value=val, points=pts.tolist())

def p_var_backbone_abs_batch(W: np.ndarray, p: float = 2.0):
    """
    Batched: W shape (M, T) → returns (values, points_list)
    """
    W = np.asarray(W, dtype=np.float64)
    M = W.shape[0]
    vals = np.empty(M, dtype=np.float64)
    points_list = []
    for i in range(M):
        vals[i], pts = _p_var_backbone_abs_numba(W[i], float(p))
        points_list.append(pts.tolist())
    return vals, points_list

import math
import numpy as np
import numba as nb
from collections import namedtuple

# ---------- core: numba-compiled backbone with Euclidean metric ----------
@nb.njit(cache=True, fastmath=True)
def _euclid(x, a, b):
    # x: (T, d)
    dsum = 0.0
    for k in range(x.shape[1]):
        diff = x[b, k] - x[a, k]
        dsum += diff * diff
    return math.sqrt(dsum)

@nb.njit(cache=True, fastmath=True)
def _p_var_backbone_euclid_numba(x, p):
    """
    Numba-accelerated version of p_var_backbone when
    path_dist(a,b) = ||x[b]-x[a]||_2 for x in R^{T x d}.
    Returns (value, points_array).
    """
    path_size = x.shape[0]  # T

    if path_size == 0:
        return -math.inf, np.empty(0, dtype=np.int64)
    if path_size == 1:
        return 0.0, np.array([0], dtype=np.int64)

    s = path_size - 1
    # N = floor(log2(s)) + 1
    N = 1
    while (s >> N) != 0:
        N += 1

    ind = np.zeros(s, dtype=np.float64)

    def ind_n(j, n):
        return (s >> n) + (j >> n)

    def ind_k(j, n):
        v = ((j >> n) << n) + (1 << (n - 1))
        return v if v < s else s  # min(…, s)

    max_p_var = 0.0
    run_p_var = np.zeros(path_size, dtype=np.float64)
    point_links = np.zeros(path_size, dtype=np.int64)

    for j in range(path_size):
        # precompute 'ind' row for this j
        for n in range(1, N + 1):
            if not ((j >> n) == (s >> n) and ((s >> (n - 1)) % 2) == 0):
                ii = ind_n(j, n)
                kk = ind_k(j, n)
                d = _euclid(x, kk, j)
                if d > ind[ii]:
                    ind[ii] = d

        if j == 0:
            continue

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
                        new_p_var = run_p_var[m] + (d ** p)
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

    # reconstruct points
    pts = np.empty(path_size, dtype=np.int64)
    cnt = 0
    point_i = s
    while True:
        pts[cnt] = point_i
        cnt += 1
        if point_i == 0:
            break
        point_i = point_links[point_i]

    # reverse first cnt entries
    for t in range(cnt // 2):
        a = t
        b = cnt - 1 - t
        pts[a], pts[b] = pts[b], pts[a]

    return run_p_var[path_size - 1], pts[:cnt]

# ---------- friendly Python wrappers ----------
_PVar = namedtuple('p_var', ['value', 'points'])

def p_var_backbone_euclid_fast(x: np.ndarray, p: float = 2.0) -> _PVar:
    """
    Drop-in replacement for p_var_backbone when dist(a,b)=||x[b]-x[a]||_2.
    x: shape (T, d)
    """
    x = np.asarray(x, dtype=np.float64)
    val, pts = _p_var_backbone_euclid_numba(x, float(p))
    return _PVar(value=val, points=pts.tolist())

def p_var_backbone_euclid_batch(W: np.ndarray, p: float = 2.0):
    """
    Batched: W shape (M, T, d) → returns (values, points_list)
    """
    W = np.asarray(W, dtype=np.float64)
    M = W.shape[0]
    vals = np.empty(M, dtype=np.float64)
    points_list = []
    for i in range(M):
        val, pts = _p_var_backbone_euclid_numba(W[i], float(p))
        vals[i] = val
        points_list.append(pts.tolist())
    return vals, points_list

import math
import numpy as np
import numba as nb

# ---------- core helpers ----------

@nb.njit(cache=True, fastmath=True)
def _euclid(x, a, b):
    # x: (T, d), Euclidean distance ||x[b]-x[a]||_2
    s = 0.0
    d = x.shape[1]
    for k in range(d):
        diff = x[b, k] - x[a, k]
        s += diff * diff
    return math.sqrt(s)

@nb.njit(cache=True, fastmath=True)
def _pvar_backbone_euclid_value_numba(x, p):
    """
    Value-only version of your backbone:
    dist(a,b) = ||x[b]-x[a]||_2, no reconstruction of points.
    x: (T, d)
    """
    T = x.shape[0]
    if T == 0:
        return -math.inf
    if T == 1:
        return 0.0

    s = T - 1
    # N = floor(log2(s)) + 1
    N = 1
    while (s >> N) != 0:
        N += 1

    ind = np.zeros(s, dtype=np.float64)

    def ind_n(j, n):
        return (s >> n) + (j >> n)

    def ind_k(j, n):
        v = ((j >> n) << n) + (1 << (n - 1))
        return v if v < s else s

    max_p_var = 0.0
    run_p_var = np.zeros(T, dtype=np.float64)

    for j in range(T):
        # update "ind" structure
        for n in range(1, N + 1):
            if not ((j >> n) == (s >> n) and ((s >> (n - 1)) % 2) == 0):
                ii = ind_n(j, n)
                kk = ind_k(j, n)
                d = _euclid(x, kk, j)
                if d > ind[ii]:
                    ind[ii] = d

        if j == 0:
            continue

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
                        new_p_var = run_p_var[m] + (d ** p)
                        if new_p_var >= max_p_var:
                            max_p_var = new_p_var
                    if m > 0:
                        while n < N and ((m >> n) % 2) == 0:
                            n += 1
                        m -= 1
                    else:
                        break

        run_p_var[j] = max_p_var

    return run_p_var[T - 1]

# ---------- full (n1 x n2) matrix with parallelism ----------
@nb.njit(parallel=True, cache=True, fastmath=True)
def pvar_backbone_matrix_numba(X1, X2, p):
    """
    X1: (n1, T, d), X2: (n2, T, d)
    Returns M[i,j] = p-var_backbone(value) of the difference path (X1[i]-X2[j]).
    """
    n1, T, d = X1.shape
    n2 = X2.shape[0]
    out = np.empty((n1, n2), dtype=np.float64)

    # Flat parallel loop over all pairs
    P = n1 * n2
    for idx in nb.prange(P):
        i = idx // n2
        j = idx - i * n2

        # work buffer for the difference path (T,d)
        diff = np.empty((T, d), dtype=np.float64)
        for t in range(T):
            for k in range(d):
                diff[t, k] = X1[i, t, k] - X2[j, t, k]

        out[i, j] = _pvar_backbone_euclid_value_numba(diff, p)
    return out

