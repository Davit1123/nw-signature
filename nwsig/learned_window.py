"""
Nadaraya-Watson classifier with a learned multi-scale signature distance
(experimental).

Representation. A path x with channels x^1, ..., x^D is mapped to z = M x with
a learned matrix M: z = A x of p = min(D, projection_dim) channels, or, if
time_channel is given, z = (x^time, A x^rest) with the time channel passed
through. On a fixed multi-scale set of windows of the time interval -- the
whole path, its halves, quarters, ..., down to 1/2^(n_scales - 1) of it, and
(shifted=True) the windows shifted by half their width -- the representation
consists of the log-signature levels 1 and 2 of z: the increments and the
signed (Levy) areas. Levels 3, ..., level of the signature of the whole path
are added; they are computed from the signature of x by the equivariance
S^k(M x) = M^{(x)k} S^k(x) of signatures under linear maps, so that the
signature of x is computed once and the learned M acts on it by a tensor
contraction.

Distance. d(x, y)^2 = sum_b w_b |F_b(x) - F_b(y)|^2 over blocks b: one block
per (scale, increments or areas) and one per signature level 3, ..., level of
the whole path, with learned weights w_b > 0. The kernel is exp(-d^2), so the
weights contain the bandwidth.

Learning. M and the weights minimise the leave-one-out Brier score
sum_c (p_c - 1{y = c})^2 of the Nadaraya-Watson classifier on the training
part (a random batch of batch_size samples per step if there are more) by
Adam. validation_fraction of the training samples are held out: the
parameters with the lowest validation Brier score are kept (early stopping).
The Brier score learns the shape of the distance; the overall scale (the
bandwidth) is then selected by cv_folds-fold cross-validation accuracy on all
training samples (bandwidth_selection='cv_accuracy'), like the bandwidth of
NWSignatureClassifier, because the Brier-optimal bandwidth is much larger than
the accuracy-optimal one: smooth class probabilities score well, but the
classifier is most accurate close to the nearest neighbour. With
bandwidth_selection='validation_brier' it minimises the Brier score on the
validation samples instead. Predictions use all training samples.

Why levels 1 and 2 on the windows: in examples/sliding_window_benchmark.py the
windows at level 1 gave most of the gain of the windows at the full level, and
levels 1 and 2 of every window follow from prefix sums of the path in O(T)
operations, differentiably in M.
"""
import time
import warnings

import iisignature as ii
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .Robnwsignature import NWSignatureClassifier

__all__ = ["multiscale_windows", "window_increments_and_areas", "NWLearnedWindowClassifier"]


# ------------------------------------------------------------------ windows

def multiscale_windows(n_timesteps, n_scales=4, shifted=True):
    """
    Multi-scale set of windows of a path with n_timesteps time points.

    Scale j = 0, ..., S - 1 consists of the 2**j windows of width 1/2**j of the
    path and, if shifted, the 2**j - 1 windows of that width shifted by half
    of it. S = n_scales, reduced so that the finest half-windows span at least
    one time step.

    Returns
    -------
    grid : ndarray of int
        Time indices of the 2**S + 1 grid points (window endpoints).
    windows : list of (i, j, scale)
        Windows [grid[i], grid[j]] and their scale, the whole path first.
    """
    if int(n_timesteps) != n_timesteps or n_timesteps < 2:
        raise ValueError(f"n_timesteps must be an integer >= 2, got {n_timesteps}.")
    if int(n_scales) != n_scales or n_scales < 1:
        raise ValueError(f"n_scales must be an integer >= 1, got {n_scales}.")
    steps = int(n_timesteps) - 1
    S = max(1, min(int(n_scales), int(np.floor(np.log2(steps))))) if steps >= 2 else 1
    G = 2 ** S if steps >= 2 else 1
    grid = np.floor(np.arange(G + 1) * steps / G + 0.5).astype(int)
    windows = []
    for scale in range(S):
        width = G // 2 ** scale
        windows += [(i * width, (i + 1) * width, scale) for i in range(2 ** scale)]
        if shifted and scale > 0:
            windows += [(i * width + width // 2, (i + 1) * width + width // 2, scale)
                        for i in range(2 ** scale - 1)]
    return grid, windows


def window_increments_and_areas(Z, grid, starts, ends):
    """
    Log-signature levels 1 and 2 of the windows [grid[starts], grid[ends]] of
    the piecewise linear paths Z, from prefix sums.

    Parameters
    ----------
    Z : torch.Tensor of shape (n, T, c)
    grid : torch.Tensor of int, time indices of the grid points
    starts, ends : torch.Tensor of int, grid indices of the windows

    Returns
    -------
    increments : torch.Tensor of shape (n, n_windows, c)
    areas : torch.Tensor of shape (n, n_windows, c (c - 1) / 2)
        Levy areas (S_ij - S_ji) / 2 of the windows for the pairs i < j of
        channels, S the level-2 signature.
    """
    n, T, c = Z.shape
    iu, ju = torch.triu_indices(c, c, offset=1, device=Z.device)
    P1 = Z - Z[:, :1]                                         # increments since the start
    dZ = Z[:, 1:] - Z[:, :-1]
    # area increments of the prefix paths: (P1_i dZ_j - P1_j dZ_i) / 2 on each step
    dA = 0.5 * (P1[:, :-1, iu] * dZ[:, :, ju] - P1[:, :-1, ju] * dZ[:, :, iu])
    prefix_area = torch.cat([torch.zeros_like(dA[:, :1]), dA.cumsum(dim=1)], dim=1)
    P1g, Ag = P1[:, grid], prefix_area[:, grid]
    P1a, P1b = P1g[:, starts], P1g[:, ends]
    increments = P1b - P1a
    # Chen: area(a, b) = area(0, b) - area(0, a) - antisym(P1(a) (x) increment(a, b))
    areas = Ag[:, ends] - Ag[:, starts] - 0.5 * (P1a[..., iu] * increments[..., ju] - P1a[..., ju] * increments[..., iu])
    return increments, areas


def _contract(S, M, k):
    """M^{(x)k} applied to the level-k signatures S of shape (n, D**k): (n, c**k)."""
    n, D = len(S), M.shape[1]
    out = S.reshape(n, *([D] * k))
    for _ in range(k):
        out = torch.tensordot(out, M, dims=([1], [1]))           # contracts the next axis, appends c
    return out.reshape(n, -1)


def _sq_dists(A, B):
    """Squared Euclidean distances between the rows of A and B (differentiable)."""
    return ((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)).clamp_min(0.0)


# --------------------------------------------------------------- estimator

class NWLearnedWindowClassifier(ClassifierMixin, BaseEstimator):
    """
    Nadaraya-Watson classifier with a learned multi-scale signature distance.

    Parameters
    ----------
    level : int or None, default=None
        Highest signature level of the whole path (levels 3..level are added
        to the windows' levels 1 and 2). None: the largest level <= 5 whose
        signatures of the original paths take at most 5e6 numbers.
    n_scales : int, default=4
        Scales of the windows: the whole path, halves, ..., 1/2**(n_scales - 1).
    shifted : bool, default=True
        Add the windows shifted by half their width.
    projection_dim : int, default=8
        Number p of learned channels (p = D if there are fewer channels D).
    time_channel : int or None, default=None
        Index of a time channel that is passed through unprojected.
    learn_projection : bool, default=True
        Learn the channel map; otherwise it stays at its initial value (the
        identity if p = D, the top-p principal directions of the increments
        otherwise) and only the weights are learned.
    max_iter : int, default=300
        Maximum number of Adam steps.
    learning_rate : float, default=0.05
        Adam step size of the log-weights.
    projection_learning_rate : float, default=0.01
        Adam step size of the channel map.
    batch_size : int, default=1024
        Training samples per step (all of the training part if fewer).
    validation_fraction : float, default=0.2
        Fraction of the training samples held out for early stopping and the
        final bandwidth.
    eval_every : int, default=10
        Steps between validation evaluations.
    patience : int, default=5
        Evaluations without improvement of the validation Brier score before
        stopping.
    bandwidth_selection : {'cv_accuracy', 'validation_brier'}, default='cv_accuracy'
        How the overall scale of the learned distance is selected at the end.
    cv_folds : int, default=5
        Folds of bandwidth_selection='cv_accuracy' (stratified and shuffled
        with random_state, as in NWSignatureClassifier).
    random_state : int, default=42
    device : str or None, default=None
        Torch device; None: 'cuda' if available, else 'cpu'.
    verbose : bool, default=False

    Attributes
    ----------
    classes_ : ndarray
    level_ : int
    projection_ : ndarray of shape (c, D)
        Learned linear map M of the channels.
    block_weights_ : dict
        Learned weight of every block: (scale, 'increments' or 'areas') for the
        windows, ('whole path', k) for the signature levels k >= 3.
    n_iter_ : int
        Adam steps taken.
    history_ : list of dict
        Training and validation Brier score (and validation accuracy) at every
        evaluation.
    cv_accuracy_ : float
        Cross-validation accuracy of the selected bandwidth (NaN with
        bandwidth_selection='validation_brier').
    fit_seconds_ : float
    """

    def __init__(self, level=None, n_scales=4, shifted=True, projection_dim=8, time_channel=None,
                 learn_projection=True, max_iter=300, learning_rate=0.05, projection_learning_rate=0.01,
                 batch_size=1024, validation_fraction=0.2, eval_every=10, patience=5,
                 bandwidth_selection="cv_accuracy", cv_folds=5, random_state=42, device=None, verbose=False):
        self.level = level
        self.n_scales = n_scales
        self.shifted = shifted
        self.projection_dim = projection_dim
        self.time_channel = time_channel
        self.learn_projection = learn_projection
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.projection_learning_rate = projection_learning_rate
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.eval_every = eval_every
        self.patience = patience
        self.bandwidth_selection = bandwidth_selection
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # ------------------------------------------------------------- helpers

    def _log(self, message):
        if self.verbose:
            print(message)

    def _validate(self):
        if self.level is not None and (int(self.level) != self.level or self.level < 2):
            raise ValueError(f"level must be None or an integer >= 2, got {self.level}")
        for name in ("n_scales", "projection_dim", "max_iter", "batch_size", "eval_every", "patience"):
            value = getattr(self, name)
            if int(value) != value or value < (0 if name == "max_iter" else 1):
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if self.batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {self.batch_size}")
        if not 0 < self.validation_fraction < 1:
            raise ValueError(f"validation_fraction must be in (0, 1), got {self.validation_fraction}")
        if self.bandwidth_selection not in ("cv_accuracy", "validation_brier"):
            raise ValueError("bandwidth_selection must be 'cv_accuracy' or 'validation_brier', "
                             f"got {self.bandwidth_selection}")
        if int(self.cv_folds) != self.cv_folds or self.cv_folds < 2:
            raise ValueError(f"cv_folds must be an integer >= 2, got {self.cv_folds}")
        if not (self.learning_rate > 0 and self.projection_learning_rate > 0):
            raise ValueError("The learning rates must be positive.")

    @staticmethod
    def _check_paths(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError("X must be a 3D array of shape (n_samples, n_timesteps, n_dimensions).")
        if X.shape[1] < 2:
            raise ValueError("The paths need at least two time points.")
        return X

    def _auto_level(self, n, D):
        level = 2
        while level < 5 and n * sum(D ** k for k in range(3, level + 2)) <= 5e6:
            level += 1
        return level

    def _high_signatures(self, X):
        """Signature levels 3..level_ of the original paths, as float32 tensors (n, D**k)."""
        if self.level_ < 3:
            return []
        S = ii.sig(np.ascontiguousarray(X), int(self.level_))
        bounds = np.cumsum([0] + [self.n_channels_ ** k for k in range(1, self.level_ + 1)])
        return [torch.as_tensor(S[:, bounds[k - 1]:bounds[k]], dtype=torch.float32, device=self.device_)
                for k in range(3, self.level_ + 1)]

    def _map(self, A):
        """Channel map M (c, D) from the learned A (p, D or D - 1)."""
        if self._time is None:
            return A
        D = self.n_channels_
        M = torch.zeros((A.shape[0] + 1, D), dtype=A.dtype, device=A.device)
        M[0, self._time] = 1.0
        M[1:, self._rest] = A
        return M

    def _blocks(self, X, high, A):
        """Unweighted feature blocks of the paths X (tensor) for the channel map from A."""
        M = self._map(A)
        increments, areas = window_increments_and_areas(X @ M.T, self._grid, self._starts, self._ends)
        blocks = []
        for sel in self._scale_index:
            blocks += [increments[:, sel].flatten(1), areas[:, sel].flatten(1)]
        blocks += [_contract(S, M, k) for k, S in zip(range(3, self.level_ + 1), high)]
        return blocks

    def _features(self, X, high, A, log_w):
        """Weighted features: |F(x) - F(y)|^2 = sum_b w_b |F_b(x) - F_b(y)|^2."""
        scale = torch.exp(0.5 * log_w)
        return torch.cat([s * b for s, b in zip(scale, self._blocks(X, high, A))], dim=1)

    @staticmethod
    def _brier(P, Y):
        return ((P - Y) ** 2).sum(dim=1).mean()

    def _proba(self, F_query, F_ref, Y_ref, chunk=2048):
        return torch.cat([torch.softmax(-_sq_dists(F_query[i:i + chunk], F_ref), dim=1) @ Y_ref
                          for i in range(0, len(F_query), chunk)])

    def _loo_brier(self, F, Y):
        eye = torch.eye(len(F), dtype=torch.bool, device=F.device)
        P = torch.softmax((-_sq_dists(F, F)).masked_fill(eye, -np.inf), dim=1) @ Y
        return self._brier(P, Y)

    # ----------------------------------------------------------------- fit

    def fit(self, X, y):
        """Learn the distance on paths X of shape (n_samples, n_timesteps, n_dimensions) and labels y."""
        start = time.time()
        self._validate()
        X = self._check_paths(X)
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)}.")
        self.device_ = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dev = self.device_
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        self._encoder = LabelEncoder().fit(y)
        self.classes_ = self._encoder.classes_
        y_int = self._encoder.transform(y)
        n, T, D = X.shape
        self.n_timesteps_, self.n_channels_ = T, D
        if self.time_channel is not None and not -D <= self.time_channel < D:
            raise ValueError(f"time_channel {self.time_channel} is not a channel of {D}-channel paths.")
        self._time = None if self.time_channel is None else int(self.time_channel) % D
        self._rest = [i for i in range(D) if i != self._time]
        self.level_ = int(self.level) if self.level is not None else self._auto_level(n, D)

        grid, windows = multiscale_windows(T, self.n_scales, self.shifted)
        self._grid = torch.as_tensor(grid, device=dev)
        self._starts = torch.as_tensor([w[0] for w in windows], device=dev)
        self._ends = torch.as_tensor([w[1] for w in windows], device=dev)
        scales = np.array([w[2] for w in windows])
        self._scale_index = [torch.as_tensor(np.nonzero(scales == s)[0], device=dev) for s in np.unique(scales)]
        self.windows_ = [(int(grid[i]), int(grid[j]), s) for i, j, s in windows]

        # training / validation split, stratified when every class has two samples
        stratify = y_int if np.bincount(y_int).min() >= 2 else None
        try:
            tr, va = train_test_split(np.arange(n), test_size=self.validation_fraction,
                                      random_state=self.random_state, stratify=stratify)
        except ValueError:      # too few samples per class for a stratified split
            tr, va = train_test_split(np.arange(n), test_size=self.validation_fraction,
                                      random_state=self.random_state)

        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        high = self._high_signatures(X)
        Y = torch.nn.functional.one_hot(torch.as_tensor(y_int, device=dev), len(self.classes_)).float()

        # initial channel map: identity, or the top principal directions of the increments
        rest = X[:, :, self._rest]
        p = min(len(self._rest), int(self.projection_dim))
        if p == len(self._rest):
            A0 = np.eye(p)
        else:
            dX = np.diff(rest[tr], axis=1).reshape(-1, len(self._rest))
            _, _, Vt = np.linalg.svd(dX - dX.mean(0), full_matrices=False)
            A0 = Vt[:p]
        A = torch.tensor(A0, dtype=torch.float32, device=dev, requires_grad=bool(self.learn_projection))

        # initial weights: every block contributes its median squared distance equally;
        # then one overall scale (the bandwidth) chosen by the LOO Brier score
        sub = torch.as_tensor(rng.permutation(tr)[:min(len(tr), self.batch_size)], device=dev)
        Ysub = Y[sub]
        with torch.no_grad():
            blocks = self._blocks(Xt[sub], [S[sub] for S in high], A)
            medians = torch.stack([_sq_dists(b, b)[~torch.eye(len(b), dtype=torch.bool, device=dev)].median()
                                   for b in blocks]).clamp_min(1e-12)
            log_w0 = -torch.log(medians * len(blocks))
            losses = {}
            for c in 10.0 ** np.arange(-3, 1.01, 0.25):
                F = self._features(Xt[sub], [S[sub] for S in high], A, log_w0 - np.log(c))
                losses[c] = float(self._loo_brier(F, Ysub))
            c0 = min(losses, key=losses.get)
        log_w = (log_w0 - np.log(c0)).clone().requires_grad_(True)

        groups = [{"params": [log_w], "lr": self.learning_rate}]
        if self.learn_projection:
            groups.append({"params": [A], "lr": self.projection_learning_rate})
        optimiser = torch.optim.Adam(groups)

        tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)

        def evaluate():
            with torch.no_grad():
                F_tr = self._features(Xt[tr_t], [S[tr_t] for S in high], A, log_w)
                F_va = self._features(Xt[va_t], [S[va_t] for S in high], A, log_w)
                P = self._proba(F_va, F_tr, Y[tr_t])
                return float(self._brier(P, Y[va_t])), float((P.argmax(1) == Y[va_t].argmax(1)).float().mean())

        best_loss, best_acc = evaluate()
        best = (log_w.detach().clone(), A.detach().clone())
        self.history_ = [dict(step=0, train=losses[c0], validation=best_loss, validation_accuracy=best_acc)]
        self._log(f"step 0: LOO Brier {losses[c0]:.4f}, validation Brier {best_loss:.4f}, accuracy {best_acc:.4f}")
        stale, step = 0, 0
        for step in range(1, int(self.max_iter) + 1):
            batch = tr_t if len(tr) <= self.batch_size else tr_t[torch.randperm(len(tr), device=dev)[:self.batch_size]]
            loss = self._loo_brier(self._features(Xt[batch], [S[batch] for S in high], A, log_w), Y[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if step % self.eval_every == 0 or step == self.max_iter:
                val_loss, val_acc = evaluate()
                self.history_.append(dict(step=step, train=loss.item(), validation=val_loss,
                                          validation_accuracy=val_acc))
                self._log(f"step {step}: LOO Brier {loss.item():.4f}, validation Brier {val_loss:.4f}, "
                          f"accuracy {val_acc:.4f}")
                if val_loss < best_loss - 1e-6:
                    best_loss, best_acc, stale = val_loss, val_acc, 0
                    best = (log_w.detach().clone(), A.detach().clone())
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        self.n_iter_ = step
        log_w, A = best

        # final bandwidth (overall scale of the distance)
        with torch.no_grad():
            if self.bandwidth_selection == "cv_accuracy":
                c, self.cv_accuracy_ = self._cv_bandwidth(self._features(Xt, high, A, log_w), y_int, Y)
            else:
                F_tr = self._features(Xt[tr_t], [S[tr_t] for S in high], A, log_w)
                F_va = self._features(Xt[va_t], [S[va_t] for S in high], A, log_w)
                factors = 10.0 ** np.arange(-1.5, 1.51, 0.125)
                scores = [float(self._brier(self._proba(F_va / np.sqrt(f), F_tr / np.sqrt(f), Y[tr_t]), Y[va_t]))
                          for f in factors]
                c, self.cv_accuracy_ = factors[int(np.argmin(scores))], float("nan")
            log_w = log_w - np.log(c)
            self._log(f"bandwidth factor {c:.3g}" + ("" if np.isnan(self.cv_accuracy_)
                                                     else f", CV accuracy {self.cv_accuracy_:.4f}"))
            self._log_w, self._A = log_w, A
            self._F_ref = self._features(Xt, high, A, log_w)
            self._Y_ref = Y

        self.projection_ = self._map(A).cpu().numpy()
        w = log_w.exp().cpu().numpy()
        names = [(int(s), kind) for s in range(len(self._scale_index)) for kind in ("increments", "areas")]
        names += [("whole path", k) for k in range(3, self.level_ + 1)]
        self.block_weights_ = dict(zip(names, w.tolist()))
        self.fit_seconds_ = time.time() - start
        return self

    def _cv_bandwidth(self, F, y_int, Y):
        """
        Factor c of the bandwidth (distances divided by c) with the highest
        K-fold CV accuracy, the folds of NWSignatureClassifier; ties to the
        smaller bandwidth.
        """
        D2 = _sq_dists(F, F)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)        # non-stratified fallback for tiny classes
            folds = NWSignatureClassifier(cv_folds=int(self.cv_folds),
                                          random_state=self.random_state)._cv_folds(y_int)
        y_t = torch.as_tensor(y_int, device=F.device)
        factors = 10.0 ** np.arange(-3.0, 1.01, 0.125)
        accuracy = np.zeros(len(factors))
        for tr, va in folds:
            tr, va = torch.as_tensor(tr, device=F.device), torch.as_tensor(va, device=F.device)
            sub, Y_tr, y_va = D2[va][:, tr], Y[tr], y_t[va]
            for i, c in enumerate(factors):
                accuracy[i] += float(((torch.softmax(-sub / c, dim=1) @ Y_tr).argmax(1) == y_va).float().mean())
        best = int(np.argmax(accuracy))
        return float(factors[best]), float(accuracy[best] / len(folds))

    # ------------------------------------------------------------- predict

    def _check_fitted(self, X):
        if not hasattr(self, "_F_ref"):
            raise NotFittedError("NWLearnedWindowClassifier is not fitted yet; call fit first.")
        X = self._check_paths(X)
        if X.shape[1:] != (self.n_timesteps_, self.n_channels_):
            raise ValueError(f"X must have {self.n_timesteps_} time points and {self.n_channels_} channels "
                             f"like the training paths, got shape {X.shape}.")
        return X

    def transform(self, X):
        """Weighted features F with |F(x) - F(y)|^2 = d(x, y)^2."""
        X = self._check_fitted(X)
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device_)
            return self._features(Xt, self._high_signatures(X), self._A, self._log_w).cpu().numpy()

    def predict_proba(self, X):
        """Kernel weight of each class (columns in the order of classes_)."""
        F = torch.as_tensor(self.transform(X), device=self.device_)
        with torch.no_grad():
            return self._proba(F, self._F_ref, self._Y_ref).cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X)        # checks that the model is fitted
        return self.classes_[np.argmax(proba, axis=1)]
