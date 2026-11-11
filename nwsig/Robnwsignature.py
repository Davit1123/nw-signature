import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
import iisignature as ii
import functools
import inspect
import warnings
from scipy.optimize import brentq
from typing import Union, Optional, Tuple, List
from .p_var_vectorized import pvar_backbone_matrix_numba


@functools.lru_cache(maxsize=None)
def _logsig_spec(dim: int, level: int):
    """
    iisignature setup for log-signatures of dim-dimensional paths.

    The default method generates machine code at runtime, which fails on some
    platforms; fall back to the equivalent method without code generation.
    """
    try:
        return ii.prepare(dim, level)
    except RuntimeError:
        return ii.prepare(dim, level, "O")


class NadarayaWatsonSignature(BaseEstimator):
    """
    Nadaraya-Watson estimator on path space, for regression and classification.

    The prediction at a path x is the kernel-weighted average of the training
    targets,

        m(x) = sum_i w_i(x) y_i,  w_i(x) = exp(-d(x, x_i)^2 / h) / sum_j exp(-d(x, x_j)^2 / h),

    and for classification the predicted class is the one with the largest
    total weight sum_{i: y_i = c} w_i(x). The distance d is the signature
    metric ||S(x) - S(y)|| between (robust) signatures truncated at
    signature_level, or one of the classical path metrics (metric_choice).
    Hyperparameters that are not fixed are selected by K-fold
    cross-validation. Inputs are raw paths or pre-computed signatures.

    Parameters
    ----------
    signature_level : int, optional (default=2)
        The signature truncation level.
    bandwidth : float, optional (default=None)
        Kernel bandwidth h in exp(-d^2 / h). If None, will be selected via
        cross-validation.
    task_type : str, optional (default='regression')
        Type of task: 'regression' or 'classification'.
    use_gpu : bool or None, optional (default=None)
        Whether to use GPU acceleration. If None, automatically detects GPU availability.
    input_type : str, optional (default='paths')
        Type of input data: 'paths' for raw path data or 'signatures' for pre-computed signatures.
    cv_folds : int, optional (default=5)
        Number of cross-validation folds for hyperparameter selection. The folds
        are shuffled (seeded by random_state) and, for classification,
        stratified by class.
    signature_mode : int, optional (default=None)
        Deprecated and ignored: every iisignature output format gives the same
        signature. Will be removed in a future version.
    log_signature : bool, optional (default=False)
        Use log-signatures instead of signatures (not with robust=True).
    bandwidth_grid : array-like, optional (default=None)
        Grid of bandwidth values for cross-validation (used if bandwidth is None).
        If None, uses [0.001, 0.005, 0.01, 0.05, 0.1, 0.5].
    signature_level_grid : array-like, optional (default=None)
        Grid of signature levels for cross-validation. If None, signature_level
        is used.
    robust : bool, optional (default=False)
        Whether to use robust signature transformations.
    robust_C : float, optional (default=None)
        Robust signature parameter C, must be >= 1. If None, will be selected via
        cross-validation.
    robust_a : float, optional (default=None)
        Robust signature parameter a, must be > 0. If None, will be selected via
        cross-validation.
    robust_C_grid : array-like, optional (default=None)
        Grid of robust_C values for cross-validation (all >= 1).
        If None, uses [1.0, 2.0, 5.0, 10.0].
    robust_a_grid : array-like, optional (default=None)
        Grid of robust_a values for cross-validation (all > 0).
        If None, uses [0.5, 1.0, 1.5, 2.0].
    random_state : int, optional (default=42)
        Seed for shuffling the cross-validation folds.
    initial_condition : bool, optional (default=False)
        Whether to include the initial condition X0 in the distance. Signatures
        are invariant to translations of the path, so by default the starting
        value is ignored. If True, X0 must be passed to fit and predict, and the
        distance becomes |x0 - y0| + rho(x, y), with rho the chosen path metric.
        X0 is used as given (it is not rescaled in robust mode).
    metric_choice : str, optional (default='signature_metric')
        Distance between paths: 'signature_metric' (Euclidean distance between
        the (robust) truncated signatures), 'supremum_metric' (sup_t |x_t - y_t|),
        'l1' (time average of |x_t - y_t|), 'l2' (root mean square of
        |x_t - y_t|), 'lp' (p-th root of the time average of |x_t - y_t|^p),
        'var1' (1-variation of x - y) or 'pvar' (p-variation of x - y). The
        path metrics compare paths on the same time grid.
    p_param : float, optional (default=2)
        Exponent p of metric_choice 'pvar' and 'lp' (p >= 1).
    verbose : bool, optional (default=False)
        Print progress (cross-validation results, selected hyperparameters).

    Attributes
    ----------
    best_bandwidth_ : float
        Bandwidth used for prediction (selected by cross-validation or given).
    best_signature_level_ : int or None
        Signature level used (None for the path metrics).
    best_robust_C_, best_robust_a_ : float or None
        Robust parameters used (None if robust=False).
    classes_ : ndarray of shape (n_classes,)
        Class labels (classification only).
    X_train_ : torch.Tensor
        Training data in the representation used by the distance: (robust)
        signatures for the signature metric, the paths otherwise.
    device_ : torch.device
        Device used for the computations.
    """

    # Grids used for hyperparameters that are not fixed and have no grid given
    _DEFAULT_BANDWIDTH_GRID = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
    _DEFAULT_ROBUST_C_GRID = (1.0, 2.0, 5.0, 10.0)
    _DEFAULT_ROBUST_A_GRID = (0.5, 1.0, 1.5, 2.0)

    def __init__(
        self,
        signature_level: int = 2,
        bandwidth: Optional[float] = None,
        task_type: str = 'regression',
        use_gpu: Optional[bool] = None,
        input_type: str = 'paths',
        cv_folds: int = 5,
        signature_mode: Optional[int] = None,
        log_signature: bool = False,
        bandwidth_grid: Optional[List[float]] = None,
        signature_level_grid: Optional[List[int]] = None,
        robust: bool = False,
        robust_C: Optional[float] = None,
        robust_a: Optional[float] = None,
        robust_C_grid: Optional[List[float]] = None,
        robust_a_grid: Optional[List[float]] = None,
        random_state: int = 42,
        initial_condition: bool = False,
        metric_choice: str = 'signature_metric',
        p_param: Optional[float] = 2,
        verbose: bool = False,
    ):
        self.signature_level = signature_level
        self.bandwidth = bandwidth
        self.task_type = task_type
        self.use_gpu = use_gpu
        self.input_type = input_type
        self.cv_folds = cv_folds
        self.signature_mode = signature_mode
        self.log_signature = log_signature
        self.bandwidth_grid = bandwidth_grid
        self.signature_level_grid = signature_level_grid
        self.robust = robust
        self.robust_C = robust_C
        self.robust_a = robust_a
        self.robust_C_grid = robust_C_grid
        self.robust_a_grid = robust_a_grid
        self.random_state = random_state
        self.initial_condition = initial_condition
        self.metric_choice = metric_choice
        self.p_param = p_param
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        """CUDA if it is available and use_gpu is not False, CPU otherwise."""
        use_cuda = torch.cuda.is_available() and self.use_gpu is not False
        return torch.device('cuda' if use_cuda else 'cpu')

    _METRICS = ('signature_metric', 'supremum_metric', 'l1', 'l2', 'lp', 'var1', 'pvar')

    def _validate_parameters(self):
        """Validate the parameters (called by fit)."""
        if self.task_type not in ['regression', 'classification']:
            raise ValueError("task_type must be 'regression' or 'classification'")

        if self.input_type not in ['paths', 'signatures']:
            raise ValueError("input_type must be 'paths' or 'signatures'")

        if self.metric_choice not in self._METRICS:
            raise ValueError(f"metric_choice must be one of {self._METRICS}, got {self.metric_choice!r}")
        if self.input_type == 'signatures' and self.metric_choice != 'signature_metric':
            raise ValueError("input_type='signatures' requires metric_choice='signature_metric'")
        if self.metric_choice in ('pvar', 'lp') and not (self.p_param is not None and self.p_param >= 1):
            raise ValueError(f"p_param must be >= 1 for metric_choice={self.metric_choice!r}, got {self.p_param}")

        if self.signature_level < 1:
            raise ValueError("signature_level must be >= 1")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")

        if self.bandwidth is not None and not self.bandwidth > 0:
            raise ValueError(f"bandwidth must be > 0, got {self.bandwidth}")
        for name, grid in [("bandwidth_grid", self.bandwidth_grid), ("signature_level_grid", self.signature_level_grid),
                           ("robust_C_grid", self.robust_C_grid), ("robust_a_grid", self.robust_a_grid)]:
            if grid is not None and len(grid) == 0:
                raise ValueError(f"{name} must not be empty")
        if self.bandwidth_grid is not None and np.any(np.asarray(self.bandwidth_grid, dtype=float) <= 0):
            raise ValueError(f"bandwidth_grid must contain only values > 0, got {self.bandwidth_grid}")
        if self.signature_level_grid is not None and any(int(l) != l or l < 1 for l in self.signature_level_grid):
            raise ValueError(f"signature_level_grid must contain integers >= 1, got {self.signature_level_grid}")

        if self.robust:
            # For C < 1 the robust signature is not defined for paths with small
            # signatures (no normalising factor exists); a must be positive.
            for name, values in [("robust_C", self.robust_C), ("robust_C_grid", self.robust_C_grid)]:
                if values is not None and np.any(np.asarray(values, dtype=float) < 1):
                    raise ValueError(f"{name} must be >= 1, got {values}")
            for name, values in [("robust_a", self.robust_a), ("robust_a_grid", self.robust_a_grid)]:
                if values is not None and np.any(np.asarray(values, dtype=float) <= 0):
                    raise ValueError(f"{name} must be > 0, got {values}")

        if self.robust and self.log_signature:
            raise ValueError(
                "robust=True cannot be combined with log_signature=True: the robust "
                "signature rescales signature level k by lambda**k, which is defined "
                "for the signature, not the log-signature."
            )

    @staticmethod
    def _signature_level_sizes(sig_dim: int, sig_level: int) -> List[int]:
        """
        Number of entries in each level 1..sig_level of a truncated signature.

        Level k of the signature of a d-dimensional path has d**k entries, so a
        signature truncated at sig_level has d + d**2 + ... + d**sig_level
        entries. The path dimension d is recovered from that length; it is
        unique because the length is strictly increasing in d.
        """
        d = 1
        while sum(d**k for k in range(1, sig_level + 1)) < sig_dim:
            d += 1
        if sum(d**k for k in range(1, sig_level + 1)) != sig_dim:
            raise ValueError(
                f"A signature with {sig_dim} entries does not correspond to truncation "
                f"level {sig_level} for any path dimension. The robust signature needs "
                "the full (not log-) signature truncated at signature_level."
            )
        return [d**k for k in range(1, sig_level + 1)]

    @staticmethod
    def _normalising_factors(level_sq: np.ndarray, C: float, a: float) -> np.ndarray:
        """
        Normalising factor lambda of the robust signature for each sample.

        level_sq[m, k-1] is the squared norm |S_k|^2 of signature level k. The
        factor solves

            sum_k lambda^(2k) |S_k|^2 = psi(|S|) - 1,   |S|^2 = 1 + sum_k |S_k|^2,

        with the function psi of Chevyrev and Oberhauser: psi(x) = x^2 for
        x^2 <= C, and C + C^(1+a) (C^(-a) - x^(-2a)) / a otherwise. Where psi is
        the identity, lambda = 1. Otherwise psi(|S|) - 1 is evaluated as
        (C - 1) + (C / a) (1 - (C / |S|^2)^a) in float64, which avoids the
        cancellation of 1 - psi(|S|) for small signatures, and lambda is the root
        in [0, 1] (it exists for C >= 1).
        """
        if not np.all(np.isfinite(level_sq)):
            raise ValueError("Robust signature: the signatures contain NaN or inf values.")

        norm_sq = 1.0 + level_sq.sum(axis=1)
        powers = 2 * np.arange(1, level_sq.shape[1] + 1)
        lam = np.ones(len(level_sq))
        for m in np.nonzero(norm_sq > C)[0]:
            target = (C - 1.0) + (C / a) * (1.0 - (C / norm_sq[m]) ** a)
            try:
                lam[m] = brentq(lambda l: np.dot(l ** powers, level_sq[m]) - target, 0.0, 1.0, xtol=1e-15)
            except ValueError as e:
                raise ValueError(
                    f"Robust signature: no normalising factor exists for sample {m}. "
                    "This happens when robust_C < 1."
                ) from e
        return lam

    def _robustsignature(self, X, sig_level, C=1.0, a_param=1.0):
        """
        Compute robust (tensor-normalised) signatures from signature tensor X.

        Each signature is rescaled by the dilation delta_lambda, which multiplies
        signature level k by lambda**k. The factor lambda is chosen per sample so
        that the result has squared norm psi(||X||):

            1 + sum_k lambda**(2k) ||X_k||^2 = psi(sqrt(1 + ||X||^2)),

        where X_k is level k (d**k entries for a d-dimensional path) and the
        leading 1 is the constant level 0.

        Parameters
        ----------
        X : torch.Tensor of shape (M, sig_dim)
            Truncated signatures, levels 1..sig_level (no constant level 0),
            in the layout returned by iisignature.sig.
        sig_level : int
            Truncation level of X.
        C : float
            Robust parameter C.
        a_param : float
            Robust parameter a.

        Returns
        -------
        Robust : torch.Tensor of shape (M, sig_dim)
            Transformed robust signatures.
        """
        # Convert to PyTorch if not already
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        M, sig_dim = X.shape
        if sig_dim == 0:
            return X.clone()  # Return input if it's empty

        device = X.device

        # Start/end index of each signature level in X
        level_sizes = self._signature_level_sizes(sig_dim, sig_level)
        bounds = np.cumsum([0] + level_sizes)

        # Steps 1-2: squared norm of each level (float64) and normalising factors
        X64 = X.detach().cpu().double().numpy()
        level_sq = np.stack([(X64[:, bounds[k]:bounds[k + 1]] ** 2).sum(axis=1) for k in range(sig_level)], axis=1)
        XX = torch.tensor(self._normalising_factors(level_sq, C, a_param), dtype=X.dtype, device=device)

        # Step 3: Apply the dilation: level k is multiplied by lambda**k
        Robust = torch.empty_like(X)
        for k in range(sig_level):
            scale_factor = XX**(k + 1)
            Robust[:, bounds[k]:bounds[k + 1]] = X[:, bounds[k]:bounds[k + 1]] * scale_factor.unsqueeze(-1)

        return Robust

    def _compute_signatures(self, paths: np.ndarray, sig_level: Optional[int] = None) -> np.ndarray:
        """
        Compute path signatures from raw path data.

        Parameters
        ----------
        paths : array-like of shape (n_samples, n_timesteps, n_dimensions)
            Raw path data.
        sig_level : int, optional
            Signature level to use. If None, uses self.signature_level.

        Returns
        -------
        signatures : array-like of shape (n_samples, signature_dim)
            Computed signatures.
        """
        if paths.ndim != 3:
            raise ValueError("Paths must be 3D array of shape (n_samples, n_timesteps, n_dimensions)")

        if sig_level is None:
            sig_level = self.signature_level

        # iisignature computes a whole batch of paths of shape (n_samples, n_timesteps, d)
        paths = np.ascontiguousarray(paths, dtype=np.float64)
        if self.log_signature:
            return ii.logsig(paths, _logsig_spec(paths.shape[-1], sig_level))
        return ii.sig(paths, sig_level)


    def _signature_distance_matrix(self, X1: torch.Tensor, X2: torch.Tensor,Y1: torch.Tensor, Y2: torch.Tensor) -> torch.Tensor:
        """
        Pairwise distances ||X1[i] - X2[j]|| between signature vectors, plus
        ||Y1[i] - Y2[j]|| between the initial conditions if initial_condition=True.

        Parameters
        ----------
        X1 : torch.Tensor of shape (n1, signature_dim)
            First set of signatures.
        X2 : torch.Tensor of shape (n2, signature_dim)
            Second set of signatures.
        Y1, Y2 : torch.Tensor of shape (n1, n_features) and (n2, n_features)
            Initial conditions.

        Returns
        -------
        distances : torch.Tensor of shape (n1, n2)
            Pairwise distances.
        """
        # This implements: np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=2)
        if self.initial_condition:
            return torch.cdist(Y1,Y2,p=2)+torch.cdist(X1, X2, p=2)
        else:
            return torch.cdist(X1, X2, p=2)


    def _pairwise_path_distance(self,
                                X1: torch.Tensor,
                                X2: torch.Tensor,
                                Y1: torch.Tensor,
                                Y2: torch.Tensor,
                                kind: str,
                                weights: Optional[torch.Tensor] = None,
                               ) -> torch.Tensor:
        """
        (n1, n2) matrix of distances between the paths X1 (n1, T, d) and X2 (n2, T, d),
        plus ||Y1[i] - Y2[j]|| between the initial conditions if initial_condition=True.

        kind (|.| is the Euclidean norm, w_t time weights, 1/T if weights is None):
          - "supremum_metric": max_t |X1[i,t] - X2[j,t]|
          - "l1":   sum_t w_t |X1[i,t] - X2[j,t]|
          - "l2":   (sum_t w_t |X1[i,t] - X2[j,t]|^2)^(1/2)
          - "lp":   (sum_t w_t |X1[i,t] - X2[j,t]|^p)^(1/p), p = p_param
          - "var1": sum_k |dX1[i,k] - dX2[j,k]|, the 1-variation of the difference path
          - "pvar": p-variation of the difference path, p = p_param
        """
        p = self.p_param
        n1, T, d = X1.shape
        n2 = X2.shape[0]
        assert X2.shape[1:] == (T, d)

        if kind == "pvar":
            # Run the fast Numba backend on CPU (float64), then map back to original device/dtype
            X1_np = X1.detach().to("cpu", torch.float64).numpy()
            X2_np = X2.detach().to("cpu", torch.float64).numpy()
            M_np  = pvar_backbone_matrix_numba(X1_np, X2_np, float(p))**(1/p)  # (n1, n2)
            out   = torch.from_numpy(M_np).to(device=X1.device, dtype=X1.dtype)

        elif kind == "var1":
            if T < 2:
                out = torch.zeros((n1, n2), device=X1.device, dtype=X1.dtype)
            else:
                dX1 = X1[:, 1:, :] - X1[:, :-1, :]
                dX2 = X2[:, 1:, :] - X2[:, :-1, :]
                A = dX1.transpose(0, 1).contiguous()   # (T-1, n1, d)
                B = dX2.transpose(0, 1).contiguous()   # (T-1, n2, d)
                D = torch.cdist(A, B, p=2)             # (T-1, n1, n2)
                out = D.sum(dim=0)                     # (n1, n2)

        else:
            A = X1.transpose(0, 1).contiguous()        # (T, n1, d)
            B = X2.transpose(0, 1).contiguous()        # (T, n2, d)
            D = torch.cdist(A, B, p=2)                 # (T, n1, n2)

            if kind == "supremum_metric":
                out = D.amax(dim=0)
            else:
                # weights
                if weights is None:
                    w = torch.full((T,), 1.0 / T, device=D.device, dtype=D.dtype)
                else:
                    w = weights.to(device=D.device, dtype=D.dtype)
                    assert w.shape == (T,)

                if kind == "l2":
                    out = torch.einsum("tij,t->ij", D.pow(2), w).clamp_min_(0).sqrt_()
                elif kind == "l1":
                    out = torch.einsum("tij,t->ij", D, w)
                elif kind == "lp":
                    out = torch.einsum("tij,t->ij", D.pow(p), w).clamp_min_(0).pow(1.0 / p)
                else:
                    raise ValueError(f"Unknown kind={kind}")

        if getattr(self, "initial_condition", False):
            out = out + torch.cdist(Y1, Y2, p=2)
        return out


    @staticmethod
    def _kernel_weights(distances: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """
        Row-normalised Gaussian kernel weights.

        Returns w[i, j] = exp(-d_ij^2 / h) / sum_k exp(-d_ik^2 / h).

        Computed as a softmax, which factors out the largest term of each row
        before exponentiating. This is algebraically the same expression, but it
        cannot underflow to 0/0 when every distance is large relative to the
        bandwidth; in that limit the weights concentrate on the nearest training
        point(s), which is the correct limit of the Nadaraya-Watson estimator.
        """
        return torch.softmax(-distances**2 / bandwidth, dim=1)

    def _nw_predict_regression(self, X_test: torch.Tensor,X0_test: torch.Tensor, X_train: torch.Tensor, X0_train: torch.Tensor,
                              y_train: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """Nadaraya-Watson regression: kernel-weighted average of y_train."""
        # Compute distance matrix: shape (n_test, n_train)
        if self.metric_choice == "signature_metric":
            distances = self._signature_distance_matrix(X_test, X_train,X0_test,X0_train)
        else:
            distances = self._pairwise_path_distance(X_test, X_train,X0_test,X0_train,kind = self.metric_choice)

        # Normalised kernel weights, shape (n_test, n_train); each row sums to 1
        weights = self._kernel_weights(distances, bandwidth)

        # Weighted prediction: sum_j w_ij * y_j
        return weights @ y_train

    def _nw_predict_classification(self, X_test: torch.Tensor,X0_test: torch.Tensor, X_train: torch.Tensor, X0_train: torch.Tensor,
                                  y_train: torch.Tensor, bandwidth: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nadaraya-Watson classification. Returns the predicted class indices and
        the class probabilities (total kernel weight of each class).
        """
        if self.metric_choice == "signature_metric":
            distances = self._signature_distance_matrix(X_test, X_train,X0_test,X0_train)
        else:
            distances = self._pairwise_path_distance(X_test, X_train,X0_test,X0_train,kind = self.metric_choice)

        # Normalised kernel weights; each row sums to 1, so the class probabilities do too
        weights = self._kernel_weights(distances, bandwidth)

        n_classes = len(self.classes_)
        pred_probs = torch.zeros((X_test.shape[0], n_classes), device=self.device_)

        # y_train holds the encoded labels 0, ..., n_classes - 1
        for idx in range(n_classes):
            indicator = (y_train == idx).float()
            pred_probs[:, idx] = torch.sum(weights * indicator.unsqueeze(0), dim=1)

        pred_classes = torch.argmax(pred_probs, dim=1)
        return pred_classes, pred_probs

    def _log(self, message: str) -> None:
        """Progress output, shown if verbose=True."""
        if self.verbose:
            print(message)

    @staticmethod
    def _describe(bandwidth, level, robust_C, robust_a) -> str:
        """Human-readable hyperparameter combination."""
        parts = [f"bandwidth={bandwidth}"]
        if level is not None:
            parts.append(f"signature level={level}")
        if robust_C is not None:
            parts.append(f"C={robust_C}, a={robust_a}")
        return ", ".join(parts)

    def _hyperparameter_candidates(self):
        """
        Candidate values of the hyperparameters.

        A value fixed by the user is its only candidate; a value left as None
        is selected from its grid (or the default grid). self is not modified.

        Returns
        -------
        levels : list
            Signature levels (``[None]`` for the path metrics).
        bandwidths : list
            Kernel bandwidths.
        robust_params : list of (C, a)
            Robust parameters (``[(None, None)]`` without robust signatures).
        """
        bandwidth_grid = self._DEFAULT_BANDWIDTH_GRID if self.bandwidth_grid is None else self.bandwidth_grid
        bandwidths = [self.bandwidth] if self.bandwidth is not None else list(bandwidth_grid)

        if self.metric_choice != "signature_metric":
            return [None], bandwidths, [(None, None)]

        # Pre-computed signatures come with a fixed level
        if self.input_type == 'signatures' or self.signature_level_grid is None:
            levels = [self.signature_level]
        else:
            levels = list(self.signature_level_grid)

        if self.robust:
            C_grid = self._DEFAULT_ROBUST_C_GRID if self.robust_C_grid is None else self.robust_C_grid
            a_grid = self._DEFAULT_ROBUST_A_GRID if self.robust_a_grid is None else self.robust_a_grid
            C_values = [self.robust_C] if self.robust_C is not None else list(C_grid)
            a_values = [self.robust_a] if self.robust_a is not None else list(a_grid)
            robust_params = [(C, a) for C in C_values for a in a_values]
        else:
            robust_params = [(None, None)]
        return levels, bandwidths, robust_params

    def _features(self, X: np.ndarray, level, robust_C, robust_a, signatures=None) -> torch.Tensor:
        """
        Representation of X used by the distance.

        For the signature metric: the signatures at the given level, robustly
        normalised with (robust_C, robust_a) if robust=True; signatures already
        computed at that level can be passed to avoid recomputing them. For the
        path metrics: the paths themselves.
        """
        if self.metric_choice != "signature_metric":
            return torch.tensor(X, dtype=torch.float32, device=self.device_)
        if signatures is None:
            signatures = self._compute_signatures(X, level) if self.input_type == 'paths' else X
        features = torch.tensor(signatures, dtype=torch.float32, device=self.device_)
        if self.robust:
            features = self._robustsignature(features, sig_level=level, C=robust_C, a_param=robust_a)
        return features

    def _cv_folds(self, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        (train, validation) index pairs of the K folds: shuffled with
        random_state, and stratified by class for classification. If every
        class has fewer than K samples, stratification is impossible and plain
        shuffled folds are used instead.
        """
        stratify = self.task_type == 'classification'
        if stratify and np.bincount(y).max() < self.cv_folds:
            warnings.warn(
                f"Every class has fewer than cv_folds={self.cv_folds} samples, so the "
                "cross-validation folds cannot be stratified by class; using "
                "non-stratified folds.",
                UserWarning, stacklevel=4,
            )
            stratify = False
        if stratify:
            splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            splitter = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        return list(splitter.split(np.zeros((len(y), 1)), y))

    def _cv_score(self, features: torch.Tensor, X0: torch.Tensor, y: np.ndarray,
                  bandwidth: float, folds) -> float:
        """
        Mean validation MSE (regression) or accuracy (classification) over the
        folds, or NaN if a fold produces non-finite predictions.
        """
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device_)
        scores = []
        for train, val in folds:
            tr, va = torch.as_tensor(train), torch.as_tensor(val)
            if self.task_type == 'regression':
                pred = self._nw_predict_regression(features[va], X0[va], features[tr], X0[tr], y_t[tr], bandwidth)
                if not torch.isfinite(pred).all():
                    return float("nan")
                scores.append(mean_squared_error(y[val], pred.cpu().numpy()))
            else:
                pred, _ = self._nw_predict_classification(features[va], X0[va], features[tr], X0[tr], y_t[tr], bandwidth)
                scores.append(accuracy_score(y[val], pred.cpu().numpy()))
        return float(np.mean(scores))

    def _cross_validate(self, X: np.ndarray, X0: np.ndarray, y: np.ndarray,
                        levels, bandwidths, robust_params) -> Tuple:
        """
        Select (bandwidth, signature level, C, a) by K-fold cross-validation.

        Every combination of the candidates is scored by its mean validation
        MSE (regression, lower is better) or accuracy (classification, higher
        is better); ties go to the combination that comes first in grid order.
        """
        folds = self._cv_folds(y)
        X0_t = torch.tensor(X0, dtype=torch.float32, device=self.device_)
        higher_is_better = self.task_type == 'classification'
        score_name = "accuracy" if higher_is_better else "MSE"
        self._log(f"Cross-validation over {len(levels) * len(bandwidths) * len(robust_params)} "
                  f"combinations, {len(folds)} folds")

        best, best_score = None, None
        for level in levels:
            signatures = None
            if self.metric_choice == "signature_metric":
                # compute the signatures once per level, reuse them for every (C, a)
                signatures = self._compute_signatures(X, level) if self.input_type == 'paths' else X
            for robust_C, robust_a in robust_params:
                features = self._features(X, level, robust_C, robust_a, signatures=signatures)
                for bandwidth in bandwidths:
                    candidate = (bandwidth, level, robust_C, robust_a)
                    score = self._cv_score(features, X0_t, y, bandwidth, folds)
                    if np.isnan(score):
                        self._log(f"  skipped {self._describe(*candidate)}: non-finite predictions")
                        continue
                    if best_score is None or (score > best_score if higher_is_better else score < best_score):
                        best, best_score = candidate, score
                        self._log(f"  new best: {self._describe(*candidate)}, CV {score_name}={score:.6f}")

        if best is None:
            raise ValueError("Cross-validation failed: every candidate produced non-finite predictions.")
        return best

    def _resolve_X0(self, X0, n_samples: int) -> np.ndarray:
        """
        Validate the initial conditions passed to fit/predict.

        With initial_condition=False, X0 is not used and a zero placeholder of
        shape (n_samples, 1) is returned, so the internal code paths stay uniform.
        """
        if not self.initial_condition:
            if X0 is not None:
                warnings.warn(
                    "X0 is ignored because initial_condition=False. Set "
                    "initial_condition=True to include it in the distance.",
                    UserWarning, stacklevel=3,
                )
            return np.zeros((n_samples, 1))
        if X0 is None:
            raise ValueError(
                "initial_condition=True requires X0, an array of shape "
                "(n_samples, n_features) with the initial condition of each sample."
            )
        X0 = np.asarray(X0, dtype=float)
        if X0.ndim == 1:
            X0 = X0[:, None]
        if X0.ndim != 2 or X0.shape[0] != n_samples:
            raise ValueError(
                f"X0 must have shape (n_samples, n_features) with n_samples={n_samples}, "
                f"got {X0.shape}"
            )
        return X0

    def fit(self, X: np.ndarray, y: np.ndarray, X0: Optional[np.ndarray] = None):
        """
        Fit the Nadaraya-Watson estimator.

        Parameters
        ----------
        X : array-like
            Training data. Shape depends on input_type:
            - If input_type='paths': shape (n_samples, n_timesteps, n_dimensions)
            - If input_type='signatures': shape (n_samples, signature_dim)
        y : array-like of shape (n_samples,)
            Target values.
        X0 : array-like of shape (n_samples, n_features) or (n_samples,), optional
            Initial conditions. Required if initial_condition=True, ignored otherwise.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_parameters()
        self.device_ = self._resolve_device()
        if self.signature_mode is not None:
            warnings.warn(
                "signature_mode is deprecated and has no effect: every iisignature output "
                "format gives the same signature. It will be removed in a future version.",
                FutureWarning, stacklevel=2,
            )

        X = np.asarray(X)
        y = np.asarray(y)
        if X0 is not None and y.ndim == 2 and np.ndim(X0) == 1:
            raise TypeError(
                "fit() takes (X, y, X0=None), but it looks like it was called with the old "
                "argument order fit(X, X0, y). Use fit(X, y), or fit(X, y, X0=X0) together "
                "with initial_condition=True."
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have different numbers of samples: {X.shape[0]} and {y.shape[0]}")
        X0 = self._resolve_X0(X0, X.shape[0])

        self._log(f"Fitting NW estimator: metric_choice='{self.metric_choice}', "
                  f"input_type='{self.input_type}', X shape {X.shape}")
        if self.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            y_fit = self.label_encoder_.fit_transform(y)
            self.classes_ = self.label_encoder_.classes_
            counts = np.bincount(y_fit)
            self._log(f"Classes: {len(counts)}, samples per class: min={counts.min()}, max={counts.max()}")
        else:
            y_fit = y
            self._log(f"Target statistics: mean={y.mean():.6f}, std={y.std():.6f}, "
                      f"range=[{y.min():.6f}, {y.max():.6f}]")

        # Cross-validate only if there is something to choose
        levels, bandwidths, robust_params = self._hyperparameter_candidates()
        if len(levels) * len(bandwidths) * len(robust_params) > 1:
            best = self._cross_validate(X, X0, y_fit, levels, bandwidths, robust_params)
        else:
            best = (bandwidths[0], levels[0], *robust_params[0])
        self.best_bandwidth_, self.best_signature_level_, self.best_robust_C_, self.best_robust_a_ = best

        self.X_train_ = self._features(X, self.best_signature_level_, self.best_robust_C_, self.best_robust_a_)
        if self.metric_choice == "signature_metric" and not torch.any(self.X_train_ != 0):
            raise ValueError("All computed signatures are zero. Check the path data format: "
                             "(n_samples, n_timesteps, n_features).")
        self.X0_train_ = torch.tensor(X0, dtype=torch.float32, device=self.device_)
        self.y_train_ = torch.tensor(y_fit, dtype=torch.float32, device=self.device_)
        self._log(f"Selected {self._describe(*best)}; training data shape {tuple(self.X_train_.shape)}")
        return self

    def _prepare_test_data(self, X: np.ndarray, X0: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Put test data into the representation the model was trained on.
        X0 must already be resolved by _resolve_X0.
        """
        X_test = self._features(X, self.best_signature_level_, self.best_robust_C_, self.best_robust_a_)
        X0_test = torch.tensor(X0, dtype=torch.float32, device=self.device_)
        return X_test, X0_test

    def predict(self, X: np.ndarray, X0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions on test data.

        Parameters
        ----------
        X : array-like
            Test data with same format as training data.
        X0 : array-like of shape (n_samples, n_features) or (n_samples,), optional
            Initial conditions of the test data. Required if initial_condition=True,
            ignored otherwise.

        Returns
        -------
        predictions : array-like of shape (n_samples,)
            Predicted values or class labels.
        """
        check_is_fitted(self, "X_train_")

        X = np.array(X)
        X0 = self._resolve_X0(X0, X.shape[0])
        X_test, X0_test = self._prepare_test_data(X, X0)

        if self.task_type == 'regression':
            predictions = self._nw_predict_regression(X_test,X0_test, self.X_train_, self.X0_train_,self.y_train_, self.best_bandwidth_)
            return predictions.cpu().numpy()
        else:
            pred_classes, _ = self._nw_predict_classification(X_test,X0_test, self.X_train_,self.X0_train_,self.y_train_, self.best_bandwidth_)
            return self.label_encoder_.inverse_transform(pred_classes.cpu().numpy())

    def predict_proba(self, X: np.ndarray, X0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Parameters
        ----------
        X : array-like
            Test data.
        X0 : array-like of shape (n_samples, n_features) or (n_samples,), optional
            Initial conditions of the test data. Required if initial_condition=True,
            ignored otherwise.

        Returns
        -------
        probabilities : array-like of shape (n_samples, n_classes)
            Predicted class probabilities. Columns follow self.classes_ (the
            sorted class labels); predict returns the label of the largest
            column.
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")

        check_is_fitted(self, "X_train_")

        X = np.array(X)
        X0 = self._resolve_X0(X0, X.shape[0])
        X_test, X0_test = self._prepare_test_data(X, X0)

        _, pred_probs = self._nw_predict_classification(X_test,X0_test, self.X_train_,self.X0_train_,self.y_train_, self.best_bandwidth_)
        return pred_probs.cpu().numpy()

    def get_params(self, deep=True):
        """Get parameters for this estimator (all arguments of __init__)."""
        names = [p for p in inspect.signature(NadarayaWatsonSignature.__init__).parameters
                 if p != 'self']
        return {name: getattr(self, name) for name in names}


class NWSignatureRegressor(RegressorMixin, NadarayaWatsonSignature):
    """Nadaraya-Watson Signature Regressor."""

    def __init__(self, **kwargs):
        kwargs['task_type'] = 'regression'
        super().__init__(**kwargs)

    def score(self, X, y, sample_weight=None, X0=None):
        """R^2 of the predictions; X0 is only needed with initial_condition=True."""
        return r2_score(y, self.predict(X, X0), sample_weight=sample_weight)


class NWSignatureClassifier(ClassifierMixin, NadarayaWatsonSignature):
    """Nadaraya-Watson Signature Classifier."""

    def __init__(self, **kwargs):
        kwargs['task_type'] = 'classification'
        super().__init__(**kwargs)

    def score(self, X, y, sample_weight=None, X0=None):
        """Accuracy of the predictions; X0 is only needed with initial_condition=True."""
        return accuracy_score(y, self.predict(X, X0), sample_weight=sample_weight)
