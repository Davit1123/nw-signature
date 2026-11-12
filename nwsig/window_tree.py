"""
Windows of the path chosen by greedy binary splitting (experimental).

WindowTree grows a binary tree of windows of the time interval, as a decision
tree grows splits of the feature space. The root is the whole path. A split
cuts a leaf window [a, b] at a time point s into [a, s] and [s, b], which are
added with a weight w; the parent is kept. The distance between paths x and y
is

    d(x, y)^2 = sum_v w_v |S_v(x) - S_v(y)|^2 / m_v,

where S_v(x) is the truncated signature of window v of the path dilated by
lambda_v = ((T - 1) / (b - a))^gamma (T time points), and m_v is the median of
|S_v(x_i) - S_v(x_j)|^2 over pairs of training paths, so that w_v is the weight
of window v relative to the root (w = 1). The windows are the same for every
path.

Each step evaluates every leaf, every cut on a grid of n_segments + 1 equally
spaced time points and every weight in weight_grid, and keeps the split that
most reduces the leave-one-out (LOO) loss of the Nadaraya-Watson estimator with
weights exp(-d^2 / h): the Brier score (default) or the cross-entropy for
classification, squared error for regression. For every candidate h is
re-selected from a coarse grid and near the current one; after a split it is
re-selected from the full grid. Growing stops when the best split reduces the
LOO loss by less than min_gain times the LOO loss of the root, or at max_depth
or max_splits. gamma is selected from gamma_grid by the LOO loss of the grown
tree.

The default criterion is the Brier score sum_c (p_c - 1{y = c})^2 rather than
the cross-entropy -log p_y: both are proper scoring rules, but the
cross-entropy is unbounded, so a few samples whose neighbours have another
class make it prefer large bandwidths (near-uniform weights), where it does
not see windows that would separate the classes locally.

The signature of a window is the tensor product of the signatures of the grid
segments it consists of (Chen's identity), so every path is read once.

WindowTree is a transformer: transform(X) returns features whose Euclidean
distances are d, to be used with NWSignatureClassifier or NWSignatureRegressor
with input_type='signatures' (robust=False).
"""
from collections import OrderedDict

import iisignature as ii
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

__all__ = ["WindowTree"]

DTYPE = torch.float64
# The LOO losses are evaluated in float32, about twice as fast as float64 and
# accurate enough to rank candidate splits; distances are computed in float64
# (|a|^2 + |b|^2 - 2ab cancels for nearby points) and then rounded.
LOO_DTYPE = torch.float32


def _level_bounds(dim, level):
    return np.cumsum([0] + [dim ** k for k in range(1, level + 1)])


def _tensor_product(A, B, dim, level):
    """
    Truncated tensor product of batches of signatures (levels 1..level, layout
    of iisignature.sig): level k of A (x) B is A_k + B_k + sum_{i<k} A_i (x) B_{k-i}.
    """
    bounds = _level_bounds(dim, level)
    n = len(A)
    out = A + B
    for k in range(2, level + 1):
        block = out[:, bounds[k - 1]:bounds[k]]
        for i in range(1, k):
            Ai = A[:, bounds[i - 1]:bounds[i]]
            Bj = B[:, bounds[k - i - 1]:bounds[k - i]]
            block += np.einsum("ni,nj->nij", Ai, Bj).reshape(n, -1)
    return out


def _dilate(S, dim, level, lam):
    """Multiply signature level k by lam**k (the signature of lam * x)."""
    if lam == 1.0:
        return S
    return S * np.repeat(lam ** np.arange(1, level + 1), np.diff(_level_bounds(dim, level)))


def _sq_dists(A, B):
    """Squared Euclidean distances between the rows of A and B, float64, built in place."""
    D2 = A @ B.T
    D2.mul_(-2.0).add_((A * A).sum(1)[:, None]).add_((B * B).sum(1)[None, :])
    return D2.clamp_min_(0.0)


class WindowTree(BaseEstimator, TransformerMixin):
    """
    Windows of the path chosen by greedy binary splitting, with the leave-one-out
    loss of the Nadaraya-Watson estimator as the splitting criterion.

    Parameters
    ----------
    level : int, default=2
        Signature truncation level of every window.
    task : {'classification', 'regression'}, default='classification'
        Classification (LOO Brier score or cross-entropy) or regression (LOO
        squared error).
    criterion : {'brier', 'cross_entropy'}, default='brier'
        LOO loss for classification.
    n_segments : int, default=32
        The cuts lie on n_segments + 1 equally spaced time points (fewer if the
        path has fewer time steps).
    max_depth : int, default=4
        Maximum depth of a window (the root has depth 0).
    max_splits : int, default=15
        Maximum number of splits.
    min_gain : float, default=0.01
        A split must reduce the LOO loss by at least min_gain times the LOO
        loss of the root (not of the current tree: near a loss of 0 every
        split would reduce the current loss by a large fraction).
    min_steps : int, default=2
        Minimum number of time steps of a window.
    weight_grid : tuple of float, default=(0.0625, 0.25, 1.0, 4.0)
        Candidate weights w of the two windows of a split.
    gamma_grid : tuple of float, default=(0.0, 0.5, 1.0)
        Candidate exponents of the dilation lambda_v = ((T - 1) / (b - a))^gamma.
        At level 1 the dilation only rescales the window, which the median
        normalisation undoes, so only gamma=0 is used.
    loo_max_samples : int, default=2000
        If there are more training samples, the LOO loss is averaged over a
        random subset of this size (predicted from all training samples).
    random_state : int, default=42
        Seed of that subset.
    verbose : bool, default=False
        Print the splits.

    Attributes
    ----------
    windows_ : list of (start, end)
        Inclusive time-index ranges of the windows, the root first.
    weights_ : ndarray
        Weight w_v of each window (1 for the root).
    depths_ : list of int
        Depth of each window.
    medians_ : ndarray
        Median m_v of the squared signature distances of each window.
    gamma_ : float
        Selected dilation exponent.
    loo_loss_init_, loo_loss_ : float
        LOO loss of the root alone and of the grown tree (each with its best h).
    loo_path_ : list of float
        LOO loss after each split.
    bandwidth_factor_ : float
        LOO-optimal h of the grown tree, relative to the median squared distance.
    gamma_losses_ : dict
        Final LOO loss for every gamma.
    """

    # bandwidth h = c * median(d^2): full grid of c, and for the candidates of a
    # split a coarse grid plus a local grid around the current c
    _FULL_FACTORS = 10.0 ** np.arange(-3.0, 2.0 + 1e-9, 0.125)
    _COARSE_FACTORS = 10.0 ** np.arange(-3.0, 2.0 + 1e-9, 0.5)
    _LOCAL_FACTORS = 10.0 ** (np.arange(-2, 3) * 0.125)
    _MEDIAN_PAIRS = 200_000
    _CACHE_BYTES = 1e9          # distance matrices of candidate windows kept between splits

    def __init__(self, level=2, task="classification", criterion="brier", n_segments=32, max_depth=4,
                 max_splits=15, min_gain=0.01, min_steps=2, weight_grid=(0.0625, 0.25, 1.0, 4.0),
                 gamma_grid=(0.0, 0.5, 1.0), loo_max_samples=2000, random_state=42, verbose=False):
        self.level = level
        self.task = task
        self.criterion = criterion
        self.n_segments = n_segments
        self.max_depth = max_depth
        self.max_splits = max_splits
        self.min_gain = min_gain
        self.min_steps = min_steps
        self.weight_grid = weight_grid
        self.gamma_grid = gamma_grid
        self.loo_max_samples = loo_max_samples
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ helpers

    def _log(self, message):
        if self.verbose:
            print(message)

    def _validate(self):
        if int(self.level) != self.level or self.level < 1:
            raise ValueError(f"level must be an integer >= 1, got {self.level}")
        if self.task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got {self.task}")
        if self.criterion not in ("brier", "cross_entropy"):
            raise ValueError(f"criterion must be 'brier' or 'cross_entropy', got {self.criterion}")
        for name in ("n_segments", "min_steps", "loo_max_samples"):
            value = getattr(self, name)
            if int(value) != value or value < (2 if name == "loo_max_samples" else 1):
                raise ValueError(f"{name} must be a positive integer, got {value}")
        for name in ("max_depth", "max_splits"):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value}")
        if not 0 <= self.min_gain < 1:
            raise ValueError(f"min_gain must be in [0, 1), got {self.min_gain}")
        if len(self.weight_grid) == 0 or min(self.weight_grid) <= 0:
            raise ValueError("weight_grid must contain positive weights")
        if len(self.gamma_grid) == 0:
            raise ValueError("gamma_grid must not be empty")

    @staticmethod
    def _check_paths(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError("X must be a 3D array of shape (n_samples, n_timesteps, n_dimensions).")
        if X.shape[1] < 2:
            raise ValueError("The paths need at least two time points.")
        return X

    def _grid(self, n_timesteps):
        steps = n_timesteps - 1
        return np.unique(np.floor(np.arange(self.n_segments + 1) * steps / self.n_segments + 0.5).astype(int))

    def _segment_signatures(self, X):
        """Signatures of the paths on each grid segment [grid[i], grid[i + 1]]."""
        return [ii.sig(np.ascontiguousarray(X[:, a:b + 1]), int(self.level))
                for a, b in zip(self.grid_[:-1], self.grid_[1:])]

    def _window_signatures(self, segments, i, j, gamma):
        """Signatures of the dilated window [grid[i], grid[j]] from the segment signatures."""
        S = segments[i]
        for k in range(i + 1, j):
            S = _tensor_product(S, segments[k], self.dim_, int(self.level))
        lam = ((self.grid_[-1] - self.grid_[0]) / (self.grid_[j] - self.grid_[i])) ** gamma
        return _dilate(S, self.dim_, int(self.level), lam)

    def _pair_median(self, D2):
        """Median of D2[r, j] over (a sample of) the pairs j != rows[r]."""
        q, n = D2.shape
        rng = np.random.default_rng(0)
        r = rng.integers(0, q, self._MEDIAN_PAIRS)
        j = (self._rows[r] + rng.integers(1, n, self._MEDIAN_PAIRS)) % n     # j != rows[r]
        return float(torch.median(D2[torch.as_tensor(r), torch.as_tensor(j)]))

    def _make_loo_loss(self, y):
        """LOO loss of the (len(rows), n) squared distances d2 and bandwidth h (weights exp(-d2 / h))."""
        rows = torch.as_tensor(self._rows)
        q, n = len(rows), len(y)
        self_mask = torch.zeros((q, n), dtype=torch.bool)
        self_mask[torch.arange(q), rows] = True

        if self.task == "regression":
            y_t = torch.as_tensor(y, dtype=LOO_DTYPE)
            y_rows = y_t[rows]

            def loss(d2, h):
                logits = (-d2.to(LOO_DTYPE) / h).masked_fill_(self_mask, -np.inf)
                return float(torch.mean((torch.softmax(logits, dim=1) @ y_t - y_rows) ** 2))
            return loss

        y_t = torch.as_tensor(y)
        if self.criterion == "brier":
            onehot = torch.nn.functional.one_hot(y_t).to(LOO_DTYPE)
            onehot_rows = onehot[rows]

            def loss(d2, h):
                logits = (-d2.to(LOO_DTYPE) / h).masked_fill_(self_mask, -np.inf)
                proba = torch.softmax(logits, dim=1) @ onehot
                return float(torch.mean(((proba - onehot_rows) ** 2).sum(dim=1)))
            return loss

        same = (y_t[rows][:, None] == y_t[None, :]) & ~self_mask
        valid = same.any(dim=1)
        if not valid.any():
            raise ValueError("The cross-entropy LOO loss needs a class with at least two samples.")
        self_mask, other_class = self_mask[valid], ~same[valid]

        def loss(d2, h):
            logits = (-d2[valid].to(LOO_DTYPE) / h).masked_fill_(self_mask, -np.inf)
            log_total = torch.logsumexp(logits, dim=1)
            log_true = torch.logsumexp(logits.masked_fill_(other_class, -np.inf), dim=1)
            return float(torch.mean(log_total - log_true))
        return loss

    def _best_bandwidth(self, D2, factors):
        """(LOO loss, c) of the best h = c * median(D2) over factors c."""
        scale = self._pair_median(D2)
        if scale <= 0:
            scale = float(D2.max()) or 1.0
        losses = [self._loo(D2, c * scale) for c in factors]
        k = int(np.nanargmin(losses))
        return losses[k], float(factors[k])

    # ------------------------------------------------------------------ growing

    def _grow(self, segments, gamma):
        """Grow the tree for one gamma; returns its windows (grid indices), weights, medians and LOO path."""
        rows = torch.as_tensor(self._rows)
        cache, medians = OrderedDict(), {}

        def node_d2(node):
            # normalised squared distances (len(rows), n) of a window, in an LRU cache
            if node in cache:
                cache.move_to_end(node)
                return cache[node]
            S = torch.as_tensor(self._window_signatures(segments, *node, gamma), dtype=DTYPE)
            D2 = _sq_dists(S[rows], S)
            medians[node] = self._pair_median(D2) or 1.0
            D2 = D2.div_(medians[node]).to(LOO_DTYPE)
            cache[node] = D2
            while len(cache) > 1 and sum(t.numel() * t.element_size() for t in cache.values()) > self._CACHE_BYTES:
                cache.popitem(last=False)
            return D2

        G = len(self.grid_) - 1
        root = (0, G)
        nodes, weights, depth = [root], [1.0], {root: 0}
        leaves = {root}
        D2 = node_d2(root).clone()
        loss, c = self._best_bandwidth(D2, self._FULL_FACTORS)
        path = [loss]
        min_improvement = self.min_gain * loss
        self._log(f"gamma={gamma}: root LOO loss {loss:.6f}")

        for _ in range(int(self.max_splits)):
            best = None
            for leaf in sorted(leaves):
                if depth[leaf] >= self.max_depth:
                    continue
                i, j = leaf
                for m in range(i + 1, j):
                    if (self.grid_[m] - self.grid_[i] < self.min_steps
                            or self.grid_[j] - self.grid_[m] < self.min_steps):
                        continue
                    children = node_d2((i, m)) + node_d2((m, j))
                    for w in self.weight_grid:
                        value, _ = self._best_bandwidth(torch.add(D2, children, alpha=w),
                                                        np.union1d(self._COARSE_FACTORS, c * self._LOCAL_FACTORS))
                        if best is None or value < best[0]:
                            best = (value, leaf, m, float(w))
            if best is None or loss - best[0] < min_improvement:
                break
            _, (i, j), m, w = best
            D2.add_(node_d2((i, m)) + node_d2((m, j)), alpha=w)
            for child in ((i, m), (m, j)):
                nodes.append(child)
                weights.append(w)
                depth[child] = depth[(i, j)] + 1
                leaves.add(child)
            leaves.discard((i, j))
            loss, c = self._best_bandwidth(D2, self._FULL_FACTORS)
            path.append(loss)
            self._log(f"  split [{self.grid_[i]}, {self.grid_[j]}] at {self.grid_[m]}, w={w}: "
                      f"LOO loss {loss:.6f}")
        return dict(nodes=nodes, weights=weights, depths=[depth[v] for v in nodes],
                    medians=[medians[v] for v in nodes], path=path, loss=loss, c=c)

    def fit(self, X, y):
        """Grow the tree on paths X of shape (n_samples, n_timesteps, n_dimensions) and targets y."""
        self._validate()
        X = self._check_paths(X)
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)}.")
        if self.task == "classification":
            _, y = np.unique(y, return_inverse=True)
        else:
            y = y.astype(np.float64)

        n = len(X)
        self.n_timesteps_, self.dim_ = X.shape[1], X.shape[2]
        self.grid_ = self._grid(self.n_timesteps_)
        if n <= self.loo_max_samples:
            self._rows = np.arange(n)
        else:
            self._rows = np.sort(np.random.default_rng(self.random_state).choice(n, self.loo_max_samples,
                                                                                 replace=False))
        self._loo = self._make_loo_loss(y)
        segments = self._segment_signatures(X)

        gammas = list(self.gamma_grid) if self.level > 1 else [0.0]
        trees = {float(g): self._grow(segments, float(g)) for g in gammas}
        self.gamma_losses_ = {g: t["loss"] for g, t in trees.items()}
        self.gamma_ = min(trees, key=lambda g: trees[g]["loss"])          # ties: the first gamma
        tree = trees[self.gamma_]

        self.windows_ = [(int(self.grid_[i]), int(self.grid_[j])) for i, j in tree["nodes"]]
        self._nodes = tree["nodes"]
        self.weights_ = np.array(tree["weights"])
        self.depths_ = tree["depths"]
        self.medians_ = np.array(tree["medians"])
        self.loo_path_ = tree["path"]
        self.loo_loss_init_, self.loo_loss_ = tree["path"][0], tree["loss"]
        self.bandwidth_factor_ = tree["c"]
        del self._loo
        return self

    def transform(self, X):
        """
        Features F of shape (n_samples, n_windows * signature_dim) with
        |F(x) - F(y)|^2 = sum_v w_v |S_v(x) - S_v(y)|^2 / m_v.
        """
        if not hasattr(self, "windows_"):
            raise NotFittedError("WindowTree is not fitted yet; call fit first.")
        X = self._check_paths(X)
        if X.shape[1:] != (self.n_timesteps_, self.dim_):
            raise ValueError(f"X must have {self.n_timesteps_} time points and {self.dim_} dimensions "
                             f"like the training paths, got shape {X.shape}.")
        segments = self._segment_signatures(X)
        blocks = [np.sqrt(w / m) * self._window_signatures(segments, i, j, self.gamma_)
                  for (i, j), w, m in zip(self._nodes, self.weights_, self.medians_)]
        return np.concatenate(blocks, axis=1)
