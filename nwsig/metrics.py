
#We define different metrics on the space of paths, which are used to define the Nadaraya-Watson estmator.
import numpy as np
import iisignature as ii
import torch


def supremum_metric(y_true, y_pred):
    """
    Compute the supremum metric (L-infinity norm) between true and predicted values.
    """
    return np.max(np.abs(y_true - y_pred))

def euclidean_metric(y_true, y_pred):
    """
    This can be used as truncated signature metric, when the input is the signature already
    """
    return np.linalg.norm(y_true - y_pred)
def trunc_sig_metric(y_true, y_pred, truncation_level=3):

    return np.linalg.norm(ii.sig(y_true, truncation_level) - ii.sig(y_pred, truncation_level))

def full_sig_metric(y_true, y_pred, static_kernel_spec='linear', sigma=1, scaling=1, subsampling_dual=1, L=1, N1=1,
                    dyadic_order=1):
    """
    Distance between two paths in the feature space of the (untruncated) signature
    kernel, sqrt(k(x, x) - 2 k(x, y) + k(y, y)), with k computed by the
    signature-kernel PDE solver of the optional sigkernel package.

    y_true, y_pred : arrays of shape (n_timesteps, n_features)
    static_kernel_spec : 'linear' or 'rbf' (with bandwidth sigma)
    dyadic_order : refinement of the PDE grid (higher is more accurate)
    scaling, subsampling_dual, L, N1 : unused, kept for backward compatibility
    """
    try:
        import sigkernel
    except ImportError as e:
        raise ImportError("full_sig_metric needs the optional 'sigkernel' package (pip install sigkernel)") from e

    if static_kernel_spec == 'linear':
        static_kernel = sigkernel.LinearKernel()
    elif static_kernel_spec == 'rbf':
        static_kernel = sigkernel.RBFKernel(sigma=sigma)
    else:
        raise ValueError("static_kernel_spec must be 'linear' or 'rbf'")
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=dyadic_order)

    # sigkernel expects batches of shape (batch, n_timesteps, n_features)
    x = torch.as_tensor(np.asarray(y_true, dtype=np.float64)[None])
    y = torch.as_tensor(np.asarray(y_pred, dtype=np.float64)[None])
    k = lambda a, b: signature_kernel.compute_kernel(a, b).item()
    dist2 = k(x, x) - 2 * k(x, y) + k(y, y)
    return np.sqrt(max(dist2, 0.0))  # clip tiny negative values from rounding
def MSE(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between true and predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)
#Todo: p-variation metric/alpha-Hölder metrc
