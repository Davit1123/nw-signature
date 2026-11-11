import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import iisignature as ii
import warnings
import scipy as sc
from typing import Union, Optional, Tuple, List
from .p_var_vectorized import pvar_backbone_matrix_numba


class NadarayaWatsonSignature(BaseEstimator):
    """
    Nadaraya-Watson estimator using path signatures for regression and classification.
    
    This estimator implements the Nadaraya-Watson kernel regression/classification
    using path signatures as features. It supports both raw path data and 
    pre-computed signatures as input.
    
    Parameters
    ----------
    signature_level : int, optional (default=2)
        The signature truncation level.
    bandwidth : float, optional (default=None)
        Kernel bandwidth parameter. If None, will be selected via cross-validation.
    task_type : str, optional (default='regression')
        Type of task: 'regression' or 'classification'.
    use_gpu : bool or None, optional (default=None)
        Whether to use GPU acceleration. If None, automatically detects GPU availability.
    input_type : str, optional (default='paths')
        Type of input data: 'paths' for raw path data or 'signatures' for pre-computed signatures.
    cv_folds : int, optional (default=5)
        Number of cross-validation folds for hyperparameter selection.
    signature_mode : int, optional (default=2)
        iisignature computation mode (1 or 2).
    bandwidth_grid : array-like, optional (default=None)
        Grid of bandwidth values for cross-validation. If None, uses default grid.
    signature_level_grid : array-like, optional (default=None)
        Grid of signature levels for cross-validation. If None, uses default grid.
    robust : bool, optional (default=False)
        Whether to use robust signature transformations.
    robust_C : float, optional (default=None)
        Robust signature parameter C. If None, will be selected via cross-validation.
    robust_a : float, optional (default=None)
        Robust signature parameter a. If None, will be selected via cross-validation.
    robust_C_grid : array-like, optional (default=None)
        Grid of robust_C values for cross-validation.
    robust_a_grid : array-like, optional (default=None)
        Grid of robust_a values for cross-validation.
    random_state : int, optional (default=42)
        Random state for reproducibility.
    """
    
    def __init__(
        self,
        signature_level: int = 2,
        bandwidth: Optional[float] = None,
        task_type: str = 'regression',
        use_gpu: Optional[bool] = None,
        input_type: str = 'paths',
        cv_folds: int = 5,
        signature_mode: int = 2,
        log_signature: bool = False,
        bandwidth_grid: Optional[List[float]] = None,
        signature_level_grid: Optional[List[int]] = None,
        robust: bool = False,
        robust_C: Optional[float] = None,
        robust_a: Optional[float] = None,
        robust_C_grid: Optional[List[float]] = None,
        robust_a_grid: Optional[List[float]] = None,
        random_state: int = 42,
        initial_condition: bool = True,
        metric_choice: str = 'signature_metric',
        p_param: Optional[float] = 2,
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
        
        # Set device
        if self.use_gpu is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Initialize attributes
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self.label_encoder_ = None
        self.best_bandwidth_ = None
        self.best_signature_level_ = None
        self.best_robust_C_ = None
        self.best_robust_a_ = None
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.task_type not in ['regression', 'classification']:
            raise ValueError("task_type must be 'regression' or 'classification'")
        
        if self.input_type not in ['paths', 'signatures']:
            raise ValueError("input_type must be 'paths' or 'signatures'")
        
        if self.signature_level < 1:
            raise ValueError("signature_level must be >= 1")
        
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")

        if self.robust and self.log_signature:
            raise ValueError(
                "robust=True cannot be combined with log_signature=True: the robust "
                "signature rescales signature level k by lambda**k, which is defined "
                "for the signature, not the log-signature."
            )
    
    def _psi_torch(self, x, C, a):
        """Vectorized psi function using PyTorch."""
        return torch.where(
            x <= torch.sqrt(torch.tensor(C, dtype=x.dtype, device=x.device)),
            x**2,
            C + C**(1+a) * (C**(-a) - x**(-2*a)) / a
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

    def _solve_polynomial_scipy(self, coeffs_np):
        """Fallback to scipy for polynomial solving."""
        M = coeffs_np.shape[0]
        roots = np.zeros(M)
        
        for m in range(M):
            try:
                roots[m] = sc.optimize.brentq(np.poly1d(coeffs_np[m][::-1]),0,10000)
            except:
                roots[m] = 1.0  # Default value on failure
                print(f"Warning: Polynomial root finding failed for index {m}")
        
        return torch.tensor(roots, dtype=torch.float32)

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

        # Step 1: polynomial in lambda, coefficients in increasing powers:
        #   constant term 1 - psi(||X||), coefficient of lambda**(2k) is ||X_k||^2
        poly_coeffs = torch.zeros((M, 2 * sig_level + 1), device=device)
        norms = torch.sqrt(1 + torch.norm(X, dim=1)**2)
        poly_coeffs[:, 0] = 1 - self._psi_torch(norms, C=C, a=a_param)
        for k in range(sig_level):
            poly_coeffs[:, 2 * (k + 1)] = torch.norm(X[:, bounds[k]:bounds[k + 1]], dim=1)**2

        # Step 2: Use scipy to solve polynomial equations
        XX = self._solve_polynomial_scipy(poly_coeffs.cpu().numpy())
        XX = XX.to(device)

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
        
        signatures = []
        
        # Compute signatures for all paths
        if self.log_signature:
            logsig_spec = ii.prepare(paths.shape[-1], sig_level)
            for i in range(paths.shape[0]):
                sig = ii.logsig(paths[i], logsig_spec)
                signatures.append(sig)
            
        else:
            for i in range(paths.shape[0]):
                sig = ii.sig(paths[i], sig_level, self.signature_mode)
                
                # Take final signature if matrix output
                if sig.ndim > 1:
                    sig = sig[-1, :]
                
                signatures.append(sig)
        
        return np.array(signatures)
            
        
        
    
    def _signature_distance_matrix(self, X1: torch.Tensor, X2: torch.Tensor,Y1: torch.Tensor, Y2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances between signature vectors.
        This matches the direct NumPy implementation exactly.
        
        Parameters
        ----------
        X1 : torch.Tensor of shape (n1, signature_dim)
            First set of signatures.
        X2 : torch.Tensor of shape (n2, signature_dim)
            Second set of signatures.
            
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
                                X1: torch.Tensor,     # (n1, T, d)
                                X2: torch.Tensor,     # (n2, T, d)
                                Y1: torch.Tensor,
                                Y2: torch.Tensor,
                                kind: str = "sup",    # "sup","l2","l1","lp","var1","pvar_backbone"
                                weights: Optional[torch.Tensor] = None,  # (T,)
                               ) -> torch.Tensor:
        """
        Returns (n1, n2) matrix of path distances between X1 and X2.
    
        kind:
          - "sup":   sup_t ||X1[i,t]-X2[j,t]||_2
          - "l2":    (sum_t w_t * ||...||_2^2)^{1/2}; if weights=None uses 1/T average
          - "l1":    sum_t w_t * ||...||_2;           if weights=None uses 1/T average
          - "lp":    (sum_t w_t * ||...||_2^p)^{1/p}
          - "var1":  sum_k ||ΔX1_k - ΔX2_k||_2   (piecewise linear 1-variation)
          - "pvar_backbone": p-variation backbone of the difference path
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
        """
        Nadaraya-Watson regression prediction.
        This matches the direct NumPy implementation exactly.
        """
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
        """Nadaraya-Watson classification prediction."""
        if self.metric_choice == "signature_metric":
            distances = self._signature_distance_matrix(X_test, X_train,X0_test,X0_train)
        else:
            distances = self._pairwise_path_distance(X_test, X_train,X0_test,X0_train,kind = self.metric_choice)
        
        # Normalised kernel weights; each row sums to 1, so the class probabilities do too
        weights = self._kernel_weights(distances, bandwidth)

        n_classes = len(self.classes_)
        pred_probs = torch.zeros((X_test.shape[0], n_classes), device=self.device)

        for idx, class_label in enumerate(self.classes_):
            indicator = (y_train == class_label).float()
            pred_probs[:, idx] = torch.sum(weights * indicator.unsqueeze(0), dim=1)

        pred_classes = torch.argmax(pred_probs, dim=1)
        return pred_classes, pred_probs
    
    def _cross_validate(self, X: np.ndarray,X0: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
        """
        Perform cross-validation to select optimal hyperparameters
        """
        # Set default grids if not provided
        if self.bandwidth_grid is None:
            self.bandwidth_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        
        # Set default robust parameter grids
        if self.robust and self.robust_C_grid is None:
            self.robust_C_grid = [0.5, 1.0, 2.0, 5.0]
        if self.robust and self.robust_a_grid is None:
            self.robust_a_grid = [0.5, 1.0, 1.5, 2.0]
        
        ## Logic for signature level CV:
        if self.input_type == 'signatures':
            # For pre-computed signatures, only CV over bandwidth (and robust params if enabled)
            self.signature_level_grid = [self.signature_level]
            print("Using pre-computed signatures, no CV on signature levels.")
        elif self.signature_level_grid is not None:
            # Use the provided signature_level_grid for CV
            pass  # self.signature_level_grid is already set
        elif self.signature_level is not None:
            # Use fixed signature_level (no CV on levels)
            self.signature_level_grid = [self.signature_level]
        elif self.signature_level is None and self.signature_level_grid is None:
            # Neither specified -> use default grid for CV
            self.signature_level_grid = [1, 2, 3]
        else:
            # Fallback case (shouldn't happen, but for safety)
            self.signature_level_grid = [2]  # or some default
        
        # Create parameter grids for robust parameters
        if self.robust:
            if self.robust_C is not None and self.robust_a is not None:
                # Fixed robust parameters
                robust_param_grid = [(self.robust_C, self.robust_a)]
                print(f"Using fixed robust parameters: C={self.robust_C}, a={self.robust_a}")
            else:
                # CV over robust parameters
                robust_param_grid = [(C, a) for C in self.robust_C_grid for a in self.robust_a_grid]
                print(f"CV over robust parameters: {len(self.robust_C_grid)} C values, {len(self.robust_a_grid)} a values")
        else:
            robust_param_grid = [(None, None)]  # No robust parameters
        
        total_combinations = (len(self.signature_level_grid) * 
                            len(self.bandwidth_grid) * 
                            len(robust_param_grid))
        
        print(f"Cross-validation: Testing {total_combinations} total parameter combinations")
        print(f"  Signature levels: {len(self.signature_level_grid)}")
        print(f"  Bandwidths: {len(self.bandwidth_grid)}")
        if self.robust:
            print(f"  Robust parameters: {len(robust_param_grid)}")
        
        best_score = np.inf  # For MSE, lower is better for regression
        best_bandwidth = self.bandwidth_grid[0]
        best_signature_level = self.signature_level_grid[0]
        best_robust_C = self.robust_C_grid[0] if self.robust and self.robust_C_grid else None
        best_robust_a = self.robust_a_grid[0] if self.robust and self.robust_a_grid else None
        total_attempts = 0
        successful_attempts = 0
        
        # Handle classification labels if needed
        if self.task_type == 'classification':
            label_encoder_cv = LabelEncoder()
            y_encoded = label_encoder_cv.fit_transform(y)
            classes_cv = np.unique(y_encoded)
            y_cv = y_encoded
            # For classification, higher accuracy is better, so we'll track the best score differently
            best_score = -np.inf
        else:
            y_cv = y
        
        for sig_level in self.signature_level_grid:
            print(f"Testing signature level {sig_level}...")
            
            # Compute signatures if needed
            if self.input_type == 'paths':
                try:
                    X_sig = self._compute_signatures(X, sig_level)
                    print(f"Computed signatures shape: {X_sig.shape}")
                    
                    # Check signature quality
                    if np.all(X_sig == 0):
                        print(f"WARNING: All signatures are zero for level {sig_level}, skipping...")
                        continue
                    if np.var(X_sig) < 1e-15:
                        print(f"WARNING: Very low signature variance for level {sig_level}: {np.var(X_sig)}")
                        
                except Exception as e:
                    print(f"Failed to compute signatures for level {sig_level}: {e}")
                    continue
            else:
                X_sig = X
            
            # Validate signature data
            if X_sig.shape[0] == 0 or X_sig.shape[1] == 0:
                print(f"Invalid signature shape: {X_sig.shape}, skipping level {sig_level}")
                continue
            
            for robust_C, robust_a in robust_param_grid:
                if self.robust:
                    print(f"  Testing robust parameters C={robust_C}, a={robust_a}")
                    # Apply robust signature transformation
                    try:
                        X_sig_robust = self._robustsignature(
                            torch.tensor(X_sig, dtype=torch.float32, device=self.device),
                            sig_level=sig_level, C=robust_C, a_param=robust_a
                        ).cpu().numpy()
                    except Exception as e:
                        print(f"    Failed to compute robust signatures: {e}")
                        continue
                else:
                    X_sig_robust = X_sig
                
                # Standard k-fold cross-validation
                n_samples = X_sig_robust.shape[0]
                fold_size = n_samples // self.cv_folds
                
                for bandwidth in self.bandwidth_grid:
                    fold_scores = []
                    
                    for fold in range(self.cv_folds):
                        total_attempts += 1
                        
                        # Define validation fold
                        val_start = fold * fold_size
                        val_end = (fold + 1) * fold_size if fold < self.cv_folds - 1 else n_samples
                        
                        # Split data
                        val_indices = np.arange(val_start, val_end)
                        train_indices = np.concatenate([np.arange(0, val_start), 
                                                       np.arange(val_end, n_samples)])
                        
                        if len(train_indices) == 0 or len(val_indices) == 0:
                            continue
                        
                        # Get fold data
                        X_train_fold = X_sig_robust[train_indices]
                        X0_train_fold = X0[train_indices]
                        X_val_fold = X_sig_robust[val_indices]
                        X0_val_fold = X0[val_indices]
                        y_train_fold = y_cv[train_indices]
                        y_val_fold = y_cv[val_indices]

                        if self.robust:
                            #normalizing the initial condition to bring it to the same scale..
                            X0_train_fold = X0_train_fold*X_train_fold[:,0][:,None]
                            X0_val_fold = X0_val_fold*X_val_fold[:,0][:,None]
                        # Validate fold data
                        if X_train_fold.shape[0] == 0 or X_val_fold.shape[0] == 0:
                            continue
                        
                        try:
                            # Convert to tensors
                            X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32, device=self.device)
                            X0_train_tensor = torch.tensor(X0_train_fold, dtype=torch.float32, device=self.device)
                            X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32, device=self.device)
                            X0_val_tensor = torch.tensor(X0_val_fold, dtype=torch.float32, device=self.device)
                            y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32, device=self.device)
                            
                            # Validate tensor shapes
                            if X_train_tensor.shape[1] != X_val_tensor.shape[1]:
                                continue
                            
                            # Make predictions
                            if self.task_type == 'regression':
                                y_pred = self._nw_predict_regression(X_val_tensor,X0_val_tensor, X_train_tensor,X0_train_tensor, y_train_tensor, bandwidth)
                                y_pred_np = y_pred.cpu().numpy()
                                
                                # Validate predictions
                                if np.any(np.isnan(y_pred_np)) or np.any(np.isinf(y_pred_np)):
                                    continue
                                
                                mse = mean_squared_error(y_val_fold, y_pred_np)
                                fold_scores.append(mse)
                                successful_attempts += 1
                            
                            else:  # Classification
                                # Set temporary classes for this fold
                                temp_classes = torch.tensor(classes_cv, device=self.device)
                                original_classes = self.classes_
                                self.classes_ = temp_classes
                                
                                y_pred_classes, _ = self._nw_predict_classification(X_val_tensor,X0_val_tensor, X_train_tensor,X0_train_tensor, y_train_tensor, bandwidth)
                                y_pred_np = y_pred_classes.cpu().numpy()
                                
                                # Validate predictions
                                if np.any(np.isnan(y_pred_np)) or np.any(np.isinf(y_pred_np)):
                                    self.classes_ = original_classes
                                    continue
                                
                                accuracy = accuracy_score(y_val_fold, y_pred_np)
                                fold_scores.append(accuracy)
                                successful_attempts += 1
                                
                                # Restore original classes
                                self.classes_ = original_classes
                                
                        except Exception as e:
                            continue
                    
                    # Process fold results
                    if len(fold_scores) > 0:
                        avg_score = np.mean(fold_scores)
                        
                        # For regression: lower MSE is better
                        # For classification: higher accuracy is better
                        if ((self.task_type == 'regression' and avg_score < best_score) or
                            (self.task_type == 'classification' and avg_score > best_score)):
                            best_score = avg_score
                            best_bandwidth = bandwidth
                            best_signature_level = sig_level
                            if self.robust:
                                best_robust_C = robust_C
                                best_robust_a = robust_a
                            
                            param_str = f"level={sig_level}, bandwidth={bandwidth}"
                            if self.robust:
                                param_str += f", C={robust_C}, a={robust_a}"
                            print(f"    New best: {param_str}, score={avg_score:.6f}")
        
        # Final validation
        print(f"\nCV Summary: {successful_attempts}/{total_attempts} successful attempts")
        
        if successful_attempts == 0:
            print("ERROR: No successful CV attempts! Using default parameters.")
            best_bandwidth = self.bandwidth_grid[0]
            best_signature_level = self.signature_level_grid[0]
            if self.robust:
                best_robust_C = self.robust_C_grid[0] if self.robust_C_grid else 1.0
                best_robust_a = self.robust_a_grid[0] if self.robust_a_grid else 1.0
        elif best_score == np.inf or best_score == -np.inf:
            print("WARNING: No improvement found in CV, using default parameters.")
            best_bandwidth = self.bandwidth_grid[0]
            best_signature_level = self.signature_level_grid[0]
            if self.robust:
                best_robust_C = self.robust_C_grid[0] if self.robust_C_grid else 1.0
                best_robust_a = self.robust_a_grid[0] if self.robust_a_grid else 1.0
        
        if self.task_type == 'regression':
            print(f"Best CV MSE: {best_score:.6f}")
        else:
            print(f"Best CV Accuracy: {best_score:.6f}")
            
        return best_bandwidth, best_signature_level, best_robust_C, best_robust_a

    def _cross_validate_non_signature(self, X: np.ndarray,X0: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
        """
        Perform cross-validation to select bandwith, with traditional norms
        """
        # Set default grids if not provided
        if self.bandwidth_grid is None:
            self.bandwidth_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        
        total_combinations = ( 
                            len(self.bandwidth_grid))
        
        print(f"Cross-validation: Testing {total_combinations} total parameter combinations")
        print(f"  Bandwidths: {len(self.bandwidth_grid)}")

        best_score = np.inf  # For MSE, lower is better for regression
        best_bandwidth = self.bandwidth_grid[0]
        total_attempts = 0
        successful_attempts = 0
        
        # Handle classification labels if needed
        if self.task_type == 'classification':
            label_encoder_cv = LabelEncoder()
            y_encoded = label_encoder_cv.fit_transform(y)
            classes_cv = np.unique(y_encoded)
            y_cv = y_encoded
            # For classification, higher accuracy is better, so we'll track the best score differently
            best_score = -np.inf
        else:
            y_cv = y
        # Standard k-fold cross-validation
        n_samples = X.shape[0]
        fold_size = n_samples // self.cv_folds
        for bandwidth in self.bandwidth_grid:
            fold_scores = []
                    
            for fold in range(self.cv_folds):
                    total_attempts += 1
                        
                    # Define validation fold
                    val_start = fold * fold_size
                    val_end = (fold + 1) * fold_size if fold < self.cv_folds - 1 else n_samples
                        
                    # Split data
                    val_indices = np.arange(val_start, val_end)
                    train_indices = np.concatenate([np.arange(0, val_start), 
                                                       np.arange(val_end, n_samples)])
                        
                    if len(train_indices) == 0 or len(val_indices) == 0:
                        continue
                        
                    # Get fold data
                    X_train_fold = X[train_indices]
                    X0_train_fold = X0[train_indices]
                    X_val_fold = X[val_indices]
                    X0_val_fold = X0[val_indices]
                    y_train_fold = y_cv[train_indices]
                    y_val_fold = y_cv[val_indices]

    
                    # Validate fold data
                    if X_train_fold.shape[0] == 0 or X_val_fold.shape[0] == 0:
                        continue
                        
                    try:
                        # Convert to tensors
                        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32, device=self.device)
                        X0_train_tensor = torch.tensor(X0_train_fold, dtype=torch.float32, device=self.device)
                        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32, device=self.device)
                        X0_val_tensor = torch.tensor(X0_val_fold, dtype=torch.float32, device=self.device)
                        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32, device=self.device)
                        # Validate tensor shapes
                        if X_train_tensor.shape[1] != X_val_tensor.shape[1]:
                            continue
                            
                        # Make predictions
                        if self.task_type == 'regression':
                            y_pred = self._nw_predict_regression(X_val_tensor,X0_val_tensor, X_train_tensor,X0_train_tensor, y_train_tensor, bandwidth)
                            y_pred_np = y_pred.cpu().numpy()
                                
                            # Validate predictions
                            if np.any(np.isnan(y_pred_np)) or np.any(np.isinf(y_pred_np)):
                                continue
                                
                            mse = mean_squared_error(y_val_fold, y_pred_np)
                            fold_scores.append(mse)
                            successful_attempts += 1
                            
                        else:  # Classification
                            # Set temporary classes for this fold
                            temp_classes = torch.tensor(classes_cv, device=self.device)
                            original_classes = self.classes_
                            self.classes_ = temp_classes
                                
                            y_pred_classes, _ = self._nw_predict_classification(X_val_tensor,X0_val_tensor, X_train_tensor,X0_train_tensor, y_train_tensor, bandwidth)
                            y_pred_np = y_pred_classes.cpu().numpy()
                                
                            # Validate predictions
                            if np.any(np.isnan(y_pred_np)) or np.any(np.isinf(y_pred_np)):
                                self.classes_ = original_classes
                                continue
                                
                            accuracy = accuracy_score(y_val_fold, y_pred_np)
                            fold_scores.append(accuracy)
                            successful_attempts += 1
                                
                            # Restore original classes
                            self.classes_ = original_classes
                                
                    except Exception as e:
                        continue
                    
            # Process fold results
            if len(fold_scores) > 0:
                avg_score = np.mean(fold_scores)

                # For regression: lower MSE is better
                # For classification: higher accuracy is better
                if ((self.task_type == 'regression' and avg_score < best_score) or
                    (self.task_type == 'classification' and avg_score > best_score)):
                    best_score = avg_score
                    best_bandwidth = bandwidth


                    param_str = f"bandwidth={bandwidth}"
                    print(f"    New best: {param_str}, score={avg_score:.6f}")
                
                
        
        # Final validation
        print(f"\nCV Summary: {successful_attempts}/{total_attempts} successful attempts")
        
        if successful_attempts == 0:
            print("ERROR: No successful CV attempts! Using default parameters.")
            best_bandwidth = self.bandwidth_grid[0]
            
        elif best_score == np.inf or best_score == -np.inf:
            print("WARNING: No improvement found in CV, using default parameters.")
            best_bandwidth = self.bandwidth_grid[0]
        
        if self.task_type == 'regression':
            print(f"Best CV MSE: {best_score:.6f}")
        else:
            print(f"Best CV Accuracy: {best_score:.6f}")
            
        return best_bandwidth
    
    def fit(self, X: np.ndarray,X0: np.ndarray, y: np.ndarray):
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
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.array(X)
        X0 = np.array(X0)
        y = np.array(y)
        
        print(f"Fitting NW estimator with input_type='{self.input_type}'")
        print(f"Input shape: {X.shape}, target shape: {y.shape}")
        print(f"Target statistics: mean={y.mean():.6f}, std={y.std():.6f}, range=[{y.min():.6f}, {y.max():.6f}]")
        
        # Handle classification labels
        if self.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y)
            self.classes_ = torch.tensor(np.unique(y_encoded), device=self.device)
            self.y_train_ = torch.tensor(y_encoded, dtype=torch.float32, device=self.device)
        else:
            self.y_train_ = torch.tensor(y, dtype=torch.float32, device=self.device)

        if self.metric_choice == "signature_metric":
            # Compute signatures if needed
            if self.input_type == 'paths':
                print("Computing signatures from paths...")
                X_sig = self._compute_signatures(X)
                
                # Check signature quality
                if np.all(X_sig == 0):
                    raise ValueError("All computed signatures are zero. Check path data format.")
                if np.var(X_sig) < 1e-12:
                    print("WARNING: Very low signature variance. Results may be poor.")
            else:
                print("Using pre-computed signatures...")
                X_sig = X
                print(f"Signature statistics: mean={X_sig.mean():.6f}, std={X_sig.std():.6f}")
            
            # Cross-validation for hyperparameter selection
            if self.bandwidth is None:
                print("Starting cross-validation for bandwidth selection...")
                self.best_bandwidth_, self.best_signature_level_,self.best_robust_C_, self.best_robust_a_ = self._cross_validate(X,X0, y)
                print(f"Cross-validation completed. Best bandwidth: {self.best_bandwidth_}")
            else:
                print(f"Using provided bandwidth: {self.bandwidth}")
                self.best_bandwidth_ = self.bandwidth
                self.best_signature_level_ = self.signature_level
                self.best_robust_C_ = self.robust_C
                self.best_robust_a_ = self.robust_a
            
            # Store final training data
            if self.input_type == 'paths':
                # Recompute with best signature level if needed
                if self.best_signature_level_ != self.signature_level:
                    X_sig = self._compute_signatures(X, self.best_signature_level_)
            
            # Apply robust signature transformation if enabled
            if self.robust:
                X_sig = self._robustsignature(
                    torch.tensor(X_sig, dtype=torch.float32, device=self.device),
                    sig_level=self.best_signature_level_,
                    C=self.best_robust_C_, a_param=self.best_robust_a_
                ).cpu().numpy()
            
            self.X_train_ = torch.tensor(X_sig, dtype=torch.float32, device=self.device)
            self.X0_train_ = torch.tensor(X0, dtype=torch.float32, device=self.device)
            if self.robust:
                self.X0_train_ = torch.tensor(X0, dtype=torch.float32, device=self.device)*X_sig[:,0][:,None]
    
            
            print(f"Training completed. Final training data shape: {self.X_train_.shape}")
            print(f"Using bandwidth: {self.best_bandwidth_}, signature level: {self.best_signature_level_}")
            if self.robust:
                print(f"Using robust parameters: C={self.best_robust_C_}, a={self.best_robust_a_}")
            return self
        else:
            if self.bandwidth is None:
                print("Starting cross-validation for bandwidth selection...")
                self.best_bandwidth_ = self._cross_validate_non_signature(X,X0, y)
                print(f"Cross-validation completed. Best bandwidth: {self.best_bandwidth_}")
            else:
                print(f"Using provided bandwidth: {self.bandwidth}")
                self.best_bandwidth_ = self.bandwidth

            self.X_train_ = torch.tensor(X, dtype=torch.float32, device=self.device)
            self.X0_train_ = torch.tensor(X0, dtype=torch.float32, device=self.device)
            
            print(f"Training completed. Final training data shape: {self.X_train_.shape}")
            print(f"Using bandwidth: {self.best_bandwidth_}")
            return self
            
            
        
    
    def predict(self, X: np.ndarray,X0: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.
        
        Parameters
        ----------
        X : array-like
            Test data with same format as training data.
            
        Returns
        -------
        predictions : array-like of shape (n_samples,)
            Predicted values or class labels.
        """
        if self.X_train_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        if self.metric_choice == "signature_metric":
            # Compute signatures if needed
            if self.input_type == 'paths':
                X_sig = self._compute_signatures(X, self.best_signature_level_)
            else:
                X_sig = X
        
            # Apply robust signature transformation if enabled
            if self.robust:
                X_sig = self._robustsignature(
                    torch.tensor(X_sig, dtype=torch.float32, device=self.device),
                    sig_level=self.best_signature_level_,
                    C=self.best_robust_C_, a_param=self.best_robust_a_
                ).cpu().numpy()
        
            X_test = torch.tensor(X_sig, dtype=torch.float32, device=self.device)
            if self.robust:
                X0_test = torch.tensor(X0, dtype=torch.float32, device=self.device)*X_test[:,0][:,None]
                #X0_test = torch.tensor(X0, dtype=torch.float32, device=self.device)
            else:
                X0_test = torch.tensor(X0, dtype=torch.float32, device=self.device)

        else:
            X_test = torch.tensor(X, dtype=torch.float32, device=self.device)
            X0_test = torch.tensor(X0, dtype=torch.float32, device=self.device)
        

        if self.task_type == 'regression':
            predictions = self._nw_predict_regression(X_test,X0_test, self.X_train_, self.X0_train_,self.y_train_, self.best_bandwidth_)
            return predictions.cpu().numpy()
        else:
            pred_classes, _ = self._nw_predict_classification(X_test,X0_test, self.X_train_,self.X0_train_,self.y_train_, self.best_bandwidth_)
            return self.label_encoder_.inverse_transform(pred_classes.cpu().numpy())
    
    def predict_proba(self, X: np.ndarray,X0: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        
        Parameters
        ----------
        X : array-like
            Test data.
            
        Returns
        -------
        probabilities : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        if self.X_train_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        # Compute signatures if needed
        if self.input_type == 'paths':
            temp_signature_level = self.signature_level
            self.signature_level = self.best_signature_level_
            X_sig = self._compute_signatures(X)
            self.signature_level = temp_signature_level
        else:
            X_sig = X
        
        X_test = torch.tensor(X_sig, dtype=torch.float32, device=self.device)
        X0_test = torch.tensor(X0, dtype=torch.float32, device=self.device)
        _, pred_probs = self._nw_predict_classification(X_test,X0_test, self.X_train_,self.X0_train_,self.y_train_, self.best_bandwidth_)
        
        return pred_probs.cpu().numpy()
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'signature_level': self.signature_level,
            'bandwidth': self.bandwidth,
            'task_type': self.task_type,
            'use_gpu': self.use_gpu,
            'input_type': self.input_type,
            'cv_folds': self.cv_folds,
            'signature_mode': self.signature_mode,
            'bandwidth_grid': self.bandwidth_grid,
            'signature_level_grid': self.signature_level_grid,
            'robust': self.robust,
            'robust_C': self.robust_C,
            'robust_a': self.robust_a,
            'robust_C_grid': self.robust_C_grid,
            'robust_a_grid': self.robust_a_grid,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class NWSignatureRegressor(NadarayaWatsonSignature, RegressorMixin):
    """Nadaraya-Watson Signature Regressor."""
    
    def __init__(self, **kwargs):
        kwargs['task_type'] = 'regression'
        super().__init__(**kwargs)


class NWSignatureClassifier(NadarayaWatsonSignature, ClassifierMixin):
    """Nadaraya-Watson Signature Classifier."""
    
    def __init__(self, **kwargs):
        kwargs['task_type'] = 'classification'
        super().__init__(**kwargs)