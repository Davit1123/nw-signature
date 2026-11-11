"""
Nadaraya-Watson estimators with a learned distance between signatures
(experimental).

The estimators use the representation z(x) of NadarayaWatsonSignature, i.e.
the (log-)signature truncated at signature_level, robustly normalised if
robust=True, and the Gaussian Nadaraya-Watson weights

    m(x) = sum_i w_i(x) y_i,   w_i(x) = exp(-d(x, x_i)^2 / (2 h^2)) / sum_j exp(-d(x, x_j)^2 / (2 h^2)),

with a distance d learned from the training data:

    distance='levels':      d_eta(x, x')^2 = sum_k exp(eta_k) |z^(k)(x) - z^(k)(x')|^2,
    distance='projection':  d_B(x, x')^2 = |B (z(x) - z(x'))|^2,

with one weight exp(eta_k) per signature level k = 1, ..., signature_level, or
a projection B with projection_dim rows.

The distance parameters minimise the leave-one-out (LOO) loss: every training
sample is predicted from the others (its own weight is masked), with squared
error for regression and cross-entropy for classification.
"""
import inspect

import iisignature as ii
import numpy as np
import torch
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

from .Robnwsignature import NadarayaWatsonSignature


class NWLearnedDistance(NadarayaWatsonSignature):
    """
    Nadaraya-Watson estimator with a distance between signatures learned by
    leave-one-out.

    Fitting has two steps.

    1. Initial model. The LOO loss of the unweighted signature metric
       (all level weights 1) is evaluated for every candidate representation
       (signature level, robust C and a) and bandwidth h; the best one is kept.
       Hyperparameters that are fixed are used as given, the others are taken
       from their grids as in NadarayaWatsonSignature.
    2. Learned distance, minimising the LOO loss with h fixed by full-batch
       Adam in float64. Rescaling the distance is the same as changing the
       bandwidth, so h is not learned separately: the learned parameters also
       contain the overall scale of the distance.

       - 'levels': starts from eta = 0 (level weights 1, the initial model).
       - 'projection': starts from the top projection_dim principal directions
         of the training features (orthonormal rows; B = 0 would be a
         stationary point), for which h is selected again by LOO. B is
         parametrised as exp(s) W / sqrt(n_features): W starts with entries of
         root mean square 1, the log-scale s at 0. Without s, Adam needs
         thousands of steps when B has to grow by orders of magnitude, e.g.
         after the directions that dominated the initial distance (and so the
         choice of h) are suppressed.

       In these units Adam changes eta and s by about learning_rate per step
       and W by about learning_rate relative to its entries, so it cannot jump
       into the flat region of very large distances, where every sample's
       weight is on its nearest neighbour. The parameters with the lowest LOO
       loss are kept, so the result is never worse than the initial distance;
       the iteration stops early once the loss has not decreased for 25 steps.

    Overfitting of the projection: B has projection_dim * n_features
    parameters, and the LOO loss can be driven to (almost) zero for any
    labels, including random ones. This always happens when the training
    features are linearly independent (typically n_features >= n_samples,
    which is common for signatures): B can then map the training samples to
    arbitrary points. It also happens with fewer features than samples: for
    100 random paths with random binary labels (39 features, projection_dim=5)
    the LOO cross-entropy goes from 0.68 to 2.5e-6. The LOO loss then no
    longer measures generalisation. The level weights (one parameter per
    level) go from 0.70 to 0.68 on the same data.

    The features of the training samples and their squared distances per level
    are computed once and cached during fitting. With max_iter=0 and
    distance='levels' the estimator is the unweighted Nadaraya-Watson estimator
    with h chosen by LOO.

    Parameters
    ----------
    distance : str, optional (default='levels')
        Learned distance: 'levels' (one weight per signature level) or
        'projection' (linear projection B of the features).
    projection_dim : int, optional (default=10)
        Number of rows of B (distance='projection'); at most the number of
        features and of training samples.
    max_iter : int, optional (default=300)
        Maximum number of Adam steps (0: no learning).
    learning_rate : float, optional (default=0.1)
        Adam learning rate (for the level weights: step size in eta).
    loo_max_samples : int, optional (default=2000)
        If there are more training samples, the LOO loss is averaged over a
        random subset of this many samples (seeded by random_state); each of
        them is still predicted from all other training samples.
    bandwidth : float, optional (default=None)
        Bandwidth h in exp(-d^2 / (2 h^2)). Note the different convention from
        NadarayaWatsonSignature, whose bandwidth h' enters as exp(-d^2 / h')
        (h' = 2 h^2). If None, h is selected by LOO.
    bandwidth_grid : array-like, optional (default=None)
        Grid of h values. If None, the grid is the median distance between
        training samples times np.geomspace(1e-3, 3, 31).
    **kwargs
        The other parameters of NadarayaWatsonSignature (signature_level,
        signature_level_grid, log_signature, robust, robust_C, robust_a, their
        grids, input_type, task_type, random_state, use_gpu, verbose) with the
        same meaning. cv_folds is not used. metric_choice must be
        'signature_metric' and initial_condition must be False.

    Attributes
    ----------
    best_bandwidth_ : float
        Bandwidth h used for prediction (convention exp(-d^2 / (2 h^2))).
    best_signature_level_, best_robust_C_, best_robust_a_
        Representation used, as in NadarayaWatsonSignature.
    level_sizes_ : list of int
        Number of features in each signature level.
    level_weights_ : ndarray of shape (signature level,)
        Learned weights exp(eta_k) (distance='levels').
    projection_ : ndarray of shape (projection_dim, n_features)
        Learned projection B (distance='projection').
    loo_loss_init_, loo_loss_ : float
        LOO loss of the initial distance (level weights 1, or the principal
        directions) and of the learned distance.
    n_iter_ : int
        Number of Adam steps performed.
    X_train_ : torch.Tensor
        Training features mapped (scaled per level, or projected by B) so that
        their Euclidean distance is the learned distance.
    """

    _DISTANCES = ('levels', 'projection')
    # Default bandwidths, relative to the median distance between training samples
    _RELATIVE_BANDWIDTH_GRID = tuple(np.geomspace(1e-3, 3.0, 31))
    # Adam stops when the LOO loss has not decreased by a relative _TOL for _PATIENCE steps
    _PATIENCE = 25
    _TOL = 1e-6

    def __init__(self, distance='levels', projection_dim=10, max_iter=300, learning_rate=0.1,
                 loo_max_samples=2000, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance
        self.projection_dim = projection_dim
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.loo_max_samples = loo_max_samples

    def get_params(self, deep=True):
        """Get parameters for this estimator (those of NadarayaWatsonSignature and its own)."""
        params = super().get_params(deep)
        own = [p for p in inspect.signature(NWLearnedDistance.__init__).parameters if p not in ('self', 'kwargs')]
        params.update({name: getattr(self, name) for name in own})
        return params

    def _validate_parameters(self):
        super()._validate_parameters()
        if self.distance not in self._DISTANCES:
            raise ValueError(f"distance must be one of {self._DISTANCES}, got {self.distance!r}")
        if self.metric_choice != 'signature_metric':
            raise ValueError("A learned distance is a distance between signatures: "
                             "metric_choice must be 'signature_metric'")
        if self.initial_condition:
            raise ValueError("initial_condition=True is not supported with a learned distance")
        if int(self.projection_dim) != self.projection_dim or self.projection_dim < 1:
            raise ValueError(f"projection_dim must be an integer >= 1, got {self.projection_dim}")
        if int(self.max_iter) != self.max_iter or self.max_iter < 0:
            raise ValueError(f"max_iter must be an integer >= 0, got {self.max_iter}")
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if int(self.loo_max_samples) != self.loo_max_samples or self.loo_max_samples < 2:
            raise ValueError(f"loo_max_samples must be an integer >= 2, got {self.loo_max_samples}")

    @staticmethod
    def _kernel_weights(distances: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """Row-normalised Gaussian weights exp(-d^2 / (2 h^2)), computed as a stable softmax."""
        return torch.softmax(-distances**2 / (2 * bandwidth**2), dim=1)

    # ------------------------------------------------------------------ features

    def _level_sizes(self, n_features: int, level: int):
        """Number of features in each level 1..level of a (log-)signature."""
        if not self.log_signature:
            return self._signature_level_sizes(n_features, level)
        # iisignature orders the log-signature (Lyndon basis) by level
        d = 1
        while ii.logsiglength(d, level) < n_features:
            d += 1
        if ii.logsiglength(d, level) != n_features:
            raise ValueError(f"A log-signature with {n_features} entries does not correspond to "
                             f"truncation level {level} for any path dimension.")
        lengths = [0] + [ii.logsiglength(d, k) for k in range(1, level + 1)]
        return [int(s) for s in np.diff(lengths)]

    def _transform(self, Z: torch.Tensor) -> torch.Tensor:
        """Map the features so that their Euclidean distance is the learned distance."""
        if self.distance == 'projection':
            return Z @ self._feature_map_
        return Z * self._feature_map_

    def _prepare_test_data(self, X, X0):
        X_test, X0_test = super()._prepare_test_data(X, X0)
        return self._transform(X_test), X0_test

    # ------------------------------------------------------------------ leave-one-out

    def _loo_rows(self, n: int) -> np.ndarray:
        """Indices of the training samples whose LOO predictions enter the loss."""
        if n <= self.loo_max_samples:
            return np.arange(n)
        rng = np.random.default_rng(self.random_state)
        return np.sort(rng.choice(n, size=self.loo_max_samples, replace=False))

    @staticmethod
    def _sq_distances(P: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
        """(len(rows), n) squared distances |P[rows[r]] - P[j]|^2 (differentiable, also at 0)."""
        P_rows = P[rows]
        sq = (P_rows ** 2).sum(dim=1)[:, None] + (P ** 2).sum(dim=1)[None, :] - 2 * P_rows @ P.T
        return sq.clamp_min(0)

    def _level_sq_distances(self, Z: torch.Tensor, rows: np.ndarray, sizes) -> torch.Tensor:
        """(levels, len(rows), n) squared distances |z^(k)(x_r) - z^(k)(x_j)|^2, float64."""
        Z = Z.to(torch.float64)
        bounds = np.cumsum([0] + list(sizes))
        blocks = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop == start:
                blocks.append(torch.zeros((len(rows), len(Z)), dtype=Z.dtype, device=Z.device))
            else:
                blocks.append(torch.cdist(Z[rows, start:stop], Z[:, start:stop]).pow(2))
        return torch.stack(blocks)

    def _make_loo_loss(self, y: np.ndarray, rows: np.ndarray):
        """
        LOO loss as a function of the squared distances d2 (len(rows), n) and h.

        Row r predicts training sample rows[r] from all training samples except
        itself. Regression: mean squared error. Classification: mean
        cross-entropy -log(total weight of the true class), computed in log
        space; samples that are the only member of their class have no
        prediction for their class and are left out.
        """
        q, n = len(rows), len(y)
        dev = self.device_
        self_mask = torch.zeros((q, n), dtype=torch.bool, device=dev)
        self_mask[torch.arange(q), torch.as_tensor(rows)] = True

        if self.task_type == 'regression':
            y_t = torch.as_tensor(y, dtype=torch.float64, device=dev)
            y_rows = y_t[rows]

            def loss(d2, h):
                logits = (-d2 / (2 * h * h)).masked_fill(self_mask, -np.inf)
                return torch.mean((torch.softmax(logits, dim=1) @ y_t - y_rows) ** 2)
            return loss

        y_t = torch.as_tensor(y, device=dev)
        same = (y_t[rows][:, None] == y_t[None, :]) & ~self_mask
        valid = same.any(dim=1)
        if not valid.any():
            raise ValueError("The cross-entropy LOO loss needs a class with at least two samples.")
        if not valid.all():
            self._log(f"  {int((~valid).sum())} samples are the only member of their class "
                      "and are left out of the LOO loss")
        self_mask, other_class = self_mask[valid], ~same[valid]

        def loss(d2, h):
            logits = (-d2[valid] / (2 * h * h)).masked_fill(self_mask, -np.inf)
            log_total = torch.logsumexp(logits, dim=1)
            log_true_class = torch.logsumexp(logits.masked_fill(other_class, -np.inf), dim=1)
            return torch.mean(log_total - log_true_class)
        return loss

    def _bandwidth_candidates(self, d2: torch.Tensor, rows: np.ndarray):
        """Fixed bandwidth, given grid, or a grid relative to the median distance."""
        if self.bandwidth is not None:
            return [self.bandwidth]
        if self.bandwidth_grid is not None:
            return list(self.bandwidth_grid)
        off_diagonal = torch.ones_like(d2, dtype=torch.bool)
        off_diagonal[torch.arange(len(rows)), torch.as_tensor(rows)] = False
        distances = d2[off_diagonal].sqrt()
        distances = distances[distances > 0]
        if len(distances) == 0:
            raise ValueError("All training samples have the same features. Check the path data "
                             "format: (n_samples, n_timesteps, n_features).")
        median = float(distances.median())
        return [median * c for c in self._RELATIVE_BANDWIDTH_GRID]

    def _initial_model(self, X, rows, loo_loss):
        """
        Representation and bandwidth with the lowest LOO loss for the unweighted
        metric. Returns (loss, h, level, C, a, features, level sizes, squared
        distances per level).
        """
        levels, _, robust_params = self._hyperparameter_candidates()
        best = None
        for level in levels:
            # compute the signatures once per level, reuse them for every (C, a)
            signatures = self._compute_signatures(X, level) if self.input_type == 'paths' else X
            for robust_C, robust_a in robust_params:
                Z = self._features(X, level, robust_C, robust_a, signatures=signatures)
                sizes = self._level_sizes(Z.shape[1], level)
                D = self._level_sq_distances(Z, rows, sizes)
                d2 = D.sum(dim=0)
                for h in self._bandwidth_candidates(d2, rows):
                    with torch.no_grad():
                        value = float(loo_loss(d2, h))
                    if np.isfinite(value) and (best is None or value < best[0]):
                        best = (value, h, level, robust_C, robust_a, Z, sizes, D)
                        self._log(f"  new best: {self._describe(h, level, robust_C, robust_a)}, "
                                  f"LOO loss={value:.6f}")
        if best is None:
            raise ValueError("Every candidate produced a non-finite LOO loss.")
        return best

    def _minimise(self, params, objective):
        """
        Minimise objective() over params (leaf tensors with requires_grad) by
        full-batch Adam, starting from their current values.

        Returns (initial loss, lowest loss, parameters with the lowest loss,
        number of steps).
        """
        snapshot = lambda: [p.detach().clone() for p in params]
        with torch.no_grad():
            initial = float(objective())
        best_value, best_params = initial, snapshot()
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        steps = since_improvement = 0
        while steps < self.max_iter and since_improvement < self._PATIENCE:
            optimizer.zero_grad()
            value = objective()
            if not torch.isfinite(value):
                break
            # value is the loss at the current parameters, before this step
            if float(value) < best_value - self._TOL * abs(best_value):
                best_value, best_params, since_improvement = float(value), snapshot(), 0
            else:
                since_improvement += 1
            value.backward()
            if not all(torch.isfinite(p.grad).all() for p in params):
                break
            optimizer.step()
            steps += 1
        with torch.no_grad():
            final = float(objective())
        if np.isfinite(final) and final < best_value:
            best_value, best_params = final, snapshot()
        return initial, best_value, best_params, steps

    def _learn_level_weights(self, D: torch.Tensor, h: float, loo_loss):
        """eta minimising the LOO loss of sum_k exp(eta_k) D_k, from eta = 0."""
        eta = torch.zeros(D.shape[0], dtype=torch.float64, device=D.device, requires_grad=True)
        initial, final, (eta,), steps = self._minimise(
            [eta], lambda: loo_loss(torch.einsum('k,kqn->qn', eta.exp(), D), h))
        return eta, initial, final, steps

    def _learn_projection(self, Z: torch.Tensor, rows: np.ndarray, loo_loss):
        """
        B minimising the LOO loss of |B (z - z')|^2, from the top principal
        directions of the training features, with h selected by LOO for them.
        Returns (B, h, initial loss, final loss, steps).
        """
        Z = Z.to(torch.float64)
        _, _, Vh = torch.linalg.svd(Z - Z.mean(dim=0), full_matrices=False)
        B0 = Vh[:min(self.projection_dim, Vh.shape[0])]
        unit = 1.0 / np.sqrt(Z.shape[1])   # root mean square of the entries of B0

        d2 = self._sq_distances(Z @ B0.T, rows)
        bandwidths = self._bandwidth_candidates(d2, rows)
        with torch.no_grad():
            losses = [float(loo_loss(d2, h)) for h in bandwidths]
        if not np.isfinite(losses).any():
            raise ValueError("Every bandwidth produced a non-finite LOO loss for the initial projection.")
        h = bandwidths[int(np.nanargmin(losses))]
        self._log(f"  initial projection: {len(B0)} principal directions, bandwidth={h}, "
                  f"LOO loss={np.nanmin(losses):.6f}")

        W = (B0 / unit).clone().requires_grad_(True)
        log_scale = torch.zeros((), dtype=Z.dtype, device=Z.device, requires_grad=True)
        projection = lambda W, log_scale: unit * log_scale.exp() * W
        initial, final, (W, log_scale), steps = self._minimise(
            [W, log_scale], lambda: loo_loss(self._sq_distances(Z @ projection(W, log_scale).T, rows), h))
        return projection(W, log_scale), h, initial, final, steps

    # ------------------------------------------------------------------ fit

    def fit(self, X, y, X0=None):
        """
        Fit the estimator and learn the distance.

        Parameters
        ----------
        X : array-like
            Paths (n_samples, n_timesteps, n_dimensions), or pre-computed
            signatures (n_samples, signature_dim) with input_type='signatures'.
        y : array-like of shape (n_samples,)
            Targets or class labels.
        X0 : ignored
            Initial conditions are not supported (a warning is issued if given).

        Returns
        -------
        self : object
        """
        self._validate_parameters()
        self.device_ = self._resolve_device()
        X, y = np.asarray(X), np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have different numbers of samples: {X.shape[0]} and {y.shape[0]}")
        self._resolve_X0(X0, X.shape[0])

        if self.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            y_fit = self.label_encoder_.fit_transform(y)
            self.classes_ = self.label_encoder_.classes_
        else:
            y_fit = y.astype(np.float64)

        rows = self._loo_rows(len(y_fit))
        loo_loss = self._make_loo_loss(y_fit, rows)
        self._log(f"Learned distance '{self.distance}': {len(y_fit)} training samples, "
                  f"LOO loss over {len(rows)} of them")

        _, h, level, robust_C, robust_a, Z, sizes, D = self._initial_model(X, rows, loo_loss)
        if self.distance == 'levels':
            eta, self.loo_loss_init_, self.loo_loss_, self.n_iter_ = self._learn_level_weights(D, h, loo_loss)
            self.level_weights_ = eta.exp().cpu().numpy()
            self._feature_map_ = torch.repeat_interleave(
                torch.as_tensor(np.sqrt(self.level_weights_), dtype=Z.dtype, device=Z.device),
                torch.as_tensor(sizes, device=Z.device))
            learned = f"level weights {np.array2string(self.level_weights_, precision=4)}"
        else:
            del D
            B, h, self.loo_loss_init_, self.loo_loss_, self.n_iter_ = self._learn_projection(Z, rows, loo_loss)
            self.projection_ = B.cpu().numpy()
            self._feature_map_ = B.T.to(Z.dtype).contiguous()
            learned = f"projection of shape {self.projection_.shape}"

        self.best_bandwidth_, self.best_signature_level_ = h, level
        self.best_robust_C_, self.best_robust_a_ = robust_C, robust_a
        self.level_sizes_ = list(sizes)
        self.X_train_ = self._transform(Z)
        self.X0_train_ = torch.zeros((len(y_fit), 1), dtype=torch.float32, device=self.device_)
        self.y_train_ = torch.tensor(y_fit, dtype=torch.float32, device=self.device_)
        self._log(f"Selected {self._describe(h, level, robust_C, robust_a)}; {learned}; LOO loss "
                  f"{self.loo_loss_init_:.6f} -> {self.loo_loss_:.6f} ({self.n_iter_} Adam steps)")
        return self


class NWLearnedDistanceRegressor(RegressorMixin, NWLearnedDistance):
    """Nadaraya-Watson regressor with a learned signature distance."""

    def __init__(self, **kwargs):
        kwargs['task_type'] = 'regression'
        super().__init__(**kwargs)


class NWLearnedDistanceClassifier(ClassifierMixin, NWLearnedDistance):
    """Nadaraya-Watson classifier with a learned signature distance."""

    def __init__(self, **kwargs):
        kwargs['task_type'] = 'classification'
        super().__init__(**kwargs)
