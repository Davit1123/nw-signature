"""
Sliding-window signatures (experimental).

The signature of the whole path is a global summary of it. Here a path x on
the grid t_0 < ... < t_{T-1} is also split into windows W_1, ..., W_J (index
ranges that share their endpoints, optionally overlapping), and every window
is represented by the truncated signature of the dilated window,

    S_j(x) = S(lambda * x|W_j) = delta_lambda S(x|W_j),

where the dilation delta_lambda multiplies signature level k by lambda**k.
The distance between two paths is

    d(x, y)^2 = |S(x) - S(y)|^2 + mu * (1/J) * sum_j |S_j(x) - S_j(y)|^2,

and include_global=False drops the first term. With mu = 0 it is the
signature metric of NadarayaWatsonSignature.

Why the dilation: level k of the signature of a path scales like its length
to the power k (divided by k!). On a window that is a fraction 1/J of the
path, the levels k >= 2 hardly contribute to |S_j(x) - S_j(y)|, and without
the dilation the windowed distance degenerates to a comparison of the
increments of the windows, i.e. of the path sampled at J + 1 points.
lambda = J restores the relative size of the levels of the whole path.

By Chen's identity, the truncated signature of the whole path is the
truncated tensor product of the truncated signatures of consecutive
non-overlapping windows, so the windows determine the signature of the whole
path at the same truncation level.

windowed_signature_features returns features whose Euclidean distances are
d(x, y), so that they can be passed to NWSignatureClassifier and
NWSignatureRegressor with input_type='signatures' (and robust=False: the
robust normalisation of the estimator would treat the features as one
signature; normalise each window here instead).
"""
import iisignature as ii
import numpy as np

from .Robnwsignature import NadarayaWatsonSignature

__all__ = ["window_edges", "sliding_window_signatures", "windowed_signature_features"]


def window_edges(n_timesteps, n_windows, overlap=0.0):
    """
    Index ranges of n_windows windows of equal length covering a path.

    The windows have width (n_timesteps - 1) / ((n_windows - 1) * (1 - overlap) + 1)
    time steps and start (1 - overlap) * width steps apart, the first at index 0
    and the last ending at index n_timesteps - 1; the endpoints are rounded to
    the grid. With overlap=0 consecutive windows share their endpoint, so
    their concatenation is the whole path.

    Parameters
    ----------
    n_timesteps : int
        Number of time points of the path (at least 2).
    n_windows : int
        Number of windows (at least 1).
    overlap : float, default=0.0
        Fraction of a window shared with the next one, in [0, 1).

    Returns
    -------
    edges : list of (start, end)
        Inclusive index ranges: window j is path[start:end + 1].
    """
    if int(n_timesteps) != n_timesteps or n_timesteps < 2:
        raise ValueError(f"n_timesteps must be an integer >= 2, got {n_timesteps}.")
    if int(n_windows) != n_windows or n_windows < 1:
        raise ValueError(f"n_windows must be an integer >= 1, got {n_windows}.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
    n_timesteps, n_windows = int(n_timesteps), int(n_windows)

    steps = n_timesteps - 1
    width = steps / ((n_windows - 1) * (1.0 - overlap) + 1.0)
    offsets = np.arange(n_windows) * width * (1.0 - overlap)
    # round half up (np.rint rounds half to even, which makes equal windows unequal)
    starts = np.floor(offsets + 0.5).astype(int)
    if overlap == 0.0:
        ends = np.append(starts[1:], steps)  # shared endpoints, exactly contiguous
    else:
        ends = np.minimum(np.floor(offsets + width + 0.5).astype(int), steps)
    ends[-1] = steps
    if np.any(ends - starts < 1):
        raise ValueError(
            f"{n_windows} windows with overlap {overlap} do not fit a path of {n_timesteps} "
            "time points: every window needs at least one time step."
        )
    return list(zip(starts.tolist(), ends.tolist()))


def _robust_normalise(S, dim, level, C, a):
    """Robust (tensor-normalised) signatures, as NadarayaWatsonSignature._robustsignature, in float64."""
    bounds = np.cumsum([0] + [dim ** k for k in range(1, level + 1)])
    level_sq = np.stack([(S[:, bounds[k]:bounds[k + 1]] ** 2).sum(axis=1) for k in range(level)], axis=1)
    lam = NadarayaWatsonSignature._normalising_factors(level_sq, C, a)
    return S * np.repeat(lam[:, None] ** np.arange(1, level + 1), np.diff(bounds), axis=1)


def _check_paths(paths, level, robust, robust_C, robust_a):
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("paths must be a 3D array of shape (n_samples, n_timesteps, n_dimensions).")
    if int(level) != level or level < 1:
        raise ValueError(f"level must be an integer >= 1, got {level}.")
    if robust:
        if robust_C is None or robust_a is None:
            raise ValueError("robust=True needs robust_C and robust_a.")
        if robust_C < 1 or robust_a <= 0:
            raise ValueError(f"robust_C must be >= 1 and robust_a > 0, got C={robust_C}, a={robust_a}.")
    return paths


def _signatures(paths, level, dilation, robust, robust_C, robust_a):
    """Signatures of dilation * paths, robustly normalised if robust."""
    S = ii.sig(np.ascontiguousarray(dilation * paths), int(level))
    if robust:
        S = _robust_normalise(S, paths.shape[-1], int(level), robust_C, robust_a)
    return S


def sliding_window_signatures(paths, level, n_windows, overlap=0.0, dilation=1.0,
                              robust=False, robust_C=None, robust_a=None):
    """
    Truncated signatures of the dilated windows of each path.

    Parameters
    ----------
    paths : array-like of shape (n_samples, n_timesteps, n_dimensions)
        Paths on a common time grid.
    level : int
        Signature truncation level.
    n_windows : int
        Number of windows (see window_edges).
    overlap : float, default=0.0
        Fraction of a window shared with the next one, in [0, 1).
    dilation : float, default=1.0
        Each window is multiplied by dilation before its signature is
        computed, which multiplies signature level k by dilation**k.
    robust : bool, default=False
        Robustly normalise the signature of each window (after the dilation)
        with robust_C >= 1 and robust_a > 0, as NadarayaWatsonSignature does.
    robust_C, robust_a : float, optional
        Parameters of the robust normalisation.

    Returns
    -------
    S : ndarray of shape (n_samples, n_windows, signature_dim)
        S[i, j] is the signature of window j of path i, levels 1..level in the
        layout of iisignature.sig; signature_dim = d + d**2 + ... + d**level.
    """
    paths = _check_paths(paths, level, robust, robust_C, robust_a)
    if not dilation > 0:
        raise ValueError(f"dilation must be positive, got {dilation}.")
    edges = window_edges(paths.shape[1], n_windows, overlap)
    return np.stack([_signatures(paths[:, a:b + 1], level, dilation, robust, robust_C, robust_a)
                     for a, b in edges], axis=1)


def windowed_signature_features(paths, level, n_windows, overlap=0.0, dilation=1.0,
                                window_weight=1.0, include_global=True,
                                robust=False, robust_C=None, robust_a=None):
    """
    Features F whose Euclidean distances are the sliding-window signature
    distance

        |F(x) - F(y)|^2 = |S(x) - S(y)|^2 + mu * (1/J) * sum_j |S_j(x) - S_j(y)|^2,

    with S the (undilated) signature of the whole path, S_j the signature of
    the dilated window j (see sliding_window_signatures), J = n_windows and
    mu = window_weight. With include_global=False the first term is dropped.

    F = [S(x), sqrt(mu / J) S_1(x), ..., sqrt(mu / J) S_J(x)] can be passed to
    NWSignatureClassifier or NWSignatureRegressor with input_type='signatures'
    and robust=False; robust=True here normalises the global signature and
    every window signature separately.

    Parameters
    ----------
    paths, level, n_windows, overlap, dilation, robust, robust_C, robust_a
        As in sliding_window_signatures.
    window_weight : float, default=1.0
        Weight mu >= 0 of the window term.
    include_global : bool, default=True
        Include the signature of the whole path.

    Returns
    -------
    F : ndarray of shape (n_samples, (include_global + n_windows) * signature_dim)
    """
    if not window_weight >= 0:
        raise ValueError(f"window_weight must be non-negative, got {window_weight}.")
    if not include_global and window_weight == 0:
        raise ValueError("window_weight=0 with include_global=False leaves no features.")
    W = sliding_window_signatures(paths, level, n_windows, overlap, dilation, robust, robust_C, robust_a)
    blocks = [np.sqrt(window_weight / n_windows) * W.reshape(len(W), -1)]
    if include_global:
        paths = np.asarray(paths, dtype=np.float64)
        blocks.insert(0, _signatures(paths, level, 1.0, robust, robust_C, robust_a))
    return np.concatenate(blocks, axis=1)
