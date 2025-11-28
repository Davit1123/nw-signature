"""
Utility functions for path simulation and data preprocessing.
"""

import numpy as np
from .FBM_package import FBM


def fBm_sampling(M, N, H, T=1.0, method="daviesharte"):
    """
    Sample fractional Brownian motion paths.

    Parameters
    ----------
    M : int
        Number of paths to simulate
    N : int
        Number of time steps
    H : float
        Hurst parameter (0 < H < 1)
    T : float, default=1.0
        Terminal time
    method : str, default="daviesharte"
        Simulation method: "daviesharte", "cholesky", or "hosking"

    Returns
    -------
    fBm : ndarray of shape (M, N+1)
        Simulated fractional Brownian motion paths
    fGn : ndarray of shape (M, N)
        Fractional Gaussian noise increments
    BM : ndarray of shape (M, N)
        Standard Brownian motion increments
    """
    fbm_gen = FBM(n=N, M=M, hurst=H, length=T, method=method)
    fBm, fGn, BM = fbm_gen.fbm()

    # FBM returns shape (N+1, M), but we want (M, N+1)
    # Transpose all outputs to match expected shape
    return fBm.T, fGn.T, BM.T


def augment_with_time(aeon_data):
    """
    Convert Aeon-style data [n_samples, dim, length] to
    [n_samples, length, dim + 1] by adding a time channel.

    Parameters
    ----------
    aeon_data : ndarray of shape (n_samples, dim, length)
        Input data in Aeon format

    Returns
    -------
    augmented_data : ndarray of shape (n_samples, length, dim + 1)
        Time-augmented data with time as the first feature
    """
    # Transpose to [n_samples, length, dim]
    data = np.transpose(aeon_data, (0, 2, 1))
    n_samples, length, dim = data.shape

    # Create normalized time vector [0, 1]
    time_vector = np.linspace(0, 1, length)
    time_channel = np.tile(time_vector, (n_samples, 1)).reshape(n_samples, length, 1)

    # Concatenate time as the first feature
    augmented_data = np.concatenate([time_channel, data], axis=2)

    return augmented_data


def suggest_truncation_level(n_obs, length, dim, max_sig_dim_total=5e6):
    """
    Suggest the maximum reasonable signature truncation level K
    given number of observations, path length, and path dimension.

    Parameters
    ----------
    n_obs : int
        Number of paths (observations)
    length : int
        Number of steps per path
    dim : int
        Path dimension (excluding time)
    max_sig_dim_total : int, default=5e6
        Maximum allowed total signature features

    Returns
    -------
    K : int
        Maximum truncation level such that total_sig_dim <= max_sig_dim_total
    """
    d = dim + 1  # time-augmented dimension
    K = 0

    while True:
        sig_dim = (d**(K + 1) - d) // (d - 1)
        total = n_obs * sig_dim
        if total > max_sig_dim_total:
            return max(K - 1, 1)  # go back one level if exceeded
        K += 1
        if K > 10:  # Safety check to avoid infinite loop
            return 10
