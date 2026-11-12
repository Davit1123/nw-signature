"""Sliding-window signatures."""
import iisignature as ii
import numpy as np
import pytest

from nwsig import (NWSignatureClassifier, sliding_window_signatures, window_edges,
                   windowed_signature_features)
from test_robust_signature import robust_reference


def scale_levels(S, dim, level, factor):
    """Multiply signature level k (dim**k entries) by factor**k."""
    sizes = [dim ** k for k in range(1, level + 1)]
    return S * np.repeat(factor ** np.arange(1, level + 1), sizes)


# ---------------------------------------------------------------- window_edges

@pytest.mark.parametrize("n_timesteps, n_windows", [(2, 1), (11, 4), (41, 8), (100, 7), (9, 8)])
def test_non_overlapping_windows_partition_the_path(n_timesteps, n_windows):
    edges = window_edges(n_timesteps, n_windows)
    assert len(edges) == n_windows
    assert edges[0][0] == 0 and edges[-1][1] == n_timesteps - 1
    for (_, end), (start, _) in zip(edges[:-1], edges[1:]):
        assert end == start                         # consecutive windows share their endpoint
    widths = [b - a for a, b in edges]
    assert min(widths) >= 1 and max(widths) - min(widths) <= 1


@pytest.mark.parametrize("n_timesteps, n_windows, overlap", [(11, 3, 0.5), (41, 15, 0.5), (100, 6, 0.25), (37, 4, 0.75)])
def test_overlapping_windows(n_timesteps, n_windows, overlap):
    edges = window_edges(n_timesteps, n_windows, overlap)
    starts, ends = np.array(edges).T
    assert len(edges) == n_windows and starts[0] == 0 and ends[-1] == n_timesteps - 1
    width = (n_timesteps - 1) / ((n_windows - 1) * (1 - overlap) + 1)
    assert np.all(np.abs((ends - starts) - width) <= 1)
    assert np.all(np.diff(starts) > 0) and np.all(starts[1:] < ends[:-1])  # each window overlaps the next


def test_half_overlap_windows_are_unions_of_half_segments():
    # 2J - 1 windows with overlap 1/2 have the width of J non-overlapping windows
    half = window_edges(41, 8)                      # 8 half-segments of 5 steps
    assert window_edges(41, 7, 0.5) == [(half[i][0], half[i + 1][1]) for i in range(7)]


def test_one_window_is_the_whole_path():
    assert window_edges(50, 1) == [(0, 49)]
    assert window_edges(50, 1, overlap=0.5) == [(0, 49)]


@pytest.mark.parametrize("args", [(1, 1), (10, 0), (10, 2.5), (10, 2, 1.0), (10, 2, -0.1), (5, 5), (5, 9, 0.5)])
def test_invalid_windows(args):
    with pytest.raises(ValueError):
        window_edges(*args)


# ---------------------------------------------------- sliding_window_signatures

def test_window_signatures_are_signatures_of_the_windows(make_paths):
    X = make_paths(5, n_timesteps=31, dim=3)
    W = sliding_window_signatures(X, 3, n_windows=4, overlap=0.5)
    edges = window_edges(31, 4, 0.5)
    assert W.shape == (5, 4, 3 + 9 + 27)
    for j, (a, b) in enumerate(edges):
        np.testing.assert_allclose(W[:, j], ii.sig(np.ascontiguousarray(X[:, a:b + 1]), 3))


def test_one_window_is_the_signature_of_the_path(make_paths):
    X = make_paths(6, dim=3)
    np.testing.assert_allclose(sliding_window_signatures(X, 4, n_windows=1)[:, 0], ii.sig(X, 4))


def tensor_product(A, B, dim, level):
    """
    Truncated tensor product of signatures in the layout of iisignature.sig:
    level k of A (x) B is A_k + B_k + sum_{i=1}^{k-1} A_i (x) B_{k-i}, in float64
    (iisignature.sigcombine computes in float32).
    """
    bounds = np.cumsum([0] + [dim ** k for k in range(1, level + 1)])
    part = lambda S, k: S[:, bounds[k - 1]:bounds[k]]
    levels = []
    for k in range(1, level + 1):
        Ck = part(A, k) + part(B, k)
        for i in range(1, k):
            Ck = Ck + np.einsum("ni,nj->nij", part(A, i), part(B, k - i)).reshape(len(A), -1)
        levels.append(Ck)
    return np.concatenate(levels, axis=1)


@pytest.mark.parametrize("n_windows", [2, 5, 8])
def test_chen_identity(make_paths, n_windows):
    # the product of the signatures of consecutive windows is the signature of the path
    X = make_paths(4, n_timesteps=41, dim=3)
    W = sliding_window_signatures(X, 3, n_windows)
    S = W[:, 0]
    for j in range(1, n_windows):
        S = tensor_product(S, W[:, j], 3, 3)
    np.testing.assert_allclose(S, ii.sig(X, 3), rtol=1e-10, atol=1e-12)


def test_dilation_multiplies_level_k_by_lambda_to_the_k(make_paths):
    X = make_paths(5, dim=3)
    W1 = sliding_window_signatures(X, 4, n_windows=3)
    W2 = sliding_window_signatures(X, 4, n_windows=3, dilation=2.5)
    np.testing.assert_allclose(W2, scale_levels(W1, 3, 4, 2.5), rtol=1e-10)


def test_robust_window_signatures(make_paths):
    X = make_paths(8, dim=3, scale=6.0)             # large signatures: the normalisation is active
    C, a = 2.0, 1.0
    W = sliding_window_signatures(X, 3, n_windows=2, dilation=2.0, robust=True, robust_C=C, robust_a=a)
    plain = sliding_window_signatures(X, 3, n_windows=2, dilation=2.0)
    assert not np.allclose(W, plain)
    for j in range(2):
        np.testing.assert_allclose(W[:, j], robust_reference(plain[:, j], 3, 3, C, a), rtol=1e-10)
    assert np.all(1 + (W ** 2).sum(axis=2) <= C * (1 + 1 / a) + 1e-9)


@pytest.mark.parametrize("kwargs", [dict(dilation=0.0), dict(robust=True), dict(robust=True, robust_C=0.5, robust_a=1.0),
                                    dict(n_windows=41)])
def test_invalid_window_signatures(make_paths, kwargs):
    kwargs = {"n_windows": 2, **kwargs}
    with pytest.raises(ValueError):
        sliding_window_signatures(make_paths(3), 2, **kwargs)


def test_paths_must_be_3d():
    with pytest.raises(ValueError):
        sliding_window_signatures(np.zeros((5, 10)), 2, 2)


# -------------------------------------------------- windowed_signature_features

@pytest.mark.parametrize("include_global", [True, False])
def test_feature_distance_is_the_windowed_distance(make_paths, include_global):
    X = make_paths(7, n_timesteps=33, dim=3)
    level, J, overlap, lam, mu = 3, 4, 0.5, 4.0, 2.5
    F = windowed_signature_features(X, level, J, overlap, dilation=lam, window_weight=mu,
                                    include_global=include_global)
    S = ii.sig(X, level)
    windows = [ii.sig(np.ascontiguousarray(lam * X[:, a:b + 1]), level) for a, b in window_edges(33, J, overlap)]
    for i in range(7):
        for k in range(7):
            expected = mu / J * sum(np.sum((w[i] - w[k]) ** 2) for w in windows)
            if include_global:
                expected += np.sum((S[i] - S[k]) ** 2)
            np.testing.assert_allclose(np.sum((F[i] - F[k]) ** 2), expected, rtol=1e-10, atol=1e-14)


def test_zero_window_weight_gives_the_signature_metric(make_paths):
    X = make_paths(6, dim=3)
    F = windowed_signature_features(X, 3, n_windows=4, window_weight=0.0)
    D = np.linalg.norm(F[:, None] - F[None], axis=2)
    S = ii.sig(X, 3)
    np.testing.assert_allclose(D, np.linalg.norm(S[:, None] - S[None], axis=2), atol=1e-12)


def test_robust_features_normalise_global_and_windows(make_paths):
    X = make_paths(6, dim=3, scale=6.0)
    C, a = 2.0, 1.5
    F = windowed_signature_features(X, 2, n_windows=3, window_weight=3.0, robust=True, robust_C=C, robust_a=a)
    sig_dim = 3 + 9
    np.testing.assert_allclose(F[:, :sig_dim], robust_reference(ii.sig(X, 2), 3, 2, C, a), rtol=1e-10)
    W = sliding_window_signatures(X, 2, 3, robust=True, robust_C=C, robust_a=a)
    np.testing.assert_allclose(F[:, sig_dim:], np.sqrt(3.0 / 3) * W.reshape(6, -1), rtol=1e-10)


@pytest.mark.parametrize("kwargs", [dict(window_weight=-1.0), dict(window_weight=0.0, include_global=False)])
def test_invalid_features(make_paths, kwargs):
    with pytest.raises(ValueError):
        windowed_signature_features(make_paths(3), 2, 2, **kwargs)


def test_features_work_with_the_estimator(classification_data):
    X, y = classification_data
    plain = NWSignatureClassifier(signature_level=2, bandwidth=0.05).fit(X, y)
    F = windowed_signature_features(X, 2, n_windows=4, window_weight=0.0)
    on_features = NWSignatureClassifier(input_type="signatures", signature_level=2, bandwidth=0.05).fit(F, y)
    np.testing.assert_array_equal(on_features.predict(F), plain.predict(X))

    F = windowed_signature_features(X, 2, n_windows=4, dilation=4.0, window_weight=1.0)
    model = NWSignatureClassifier(input_type="signatures", signature_level=2).fit(F, y)
    assert model.score(F, y) > 0.5
