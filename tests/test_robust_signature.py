"""The robust (tensor-normalised) signature against its definition."""
import iisignature as ii
import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from nwsig import NadarayaWatsonSignature


def psi(x, C, a):
    return x ** 2 if x <= np.sqrt(C) else C + C ** (1 + a) * (C ** (-a) - x ** (-2 * a)) / a


def robust_reference(S, dim, level, C, a):
    """
    Phi(x) = delta_lambda S(x): level k of the signature (dim**k entries) is
    multiplied by lambda**k, where 1 + sum_k lambda**(2k) |S_k|^2 = psi(|S|),
    |S|^2 = 1 + sum_k |S_k|^2.
    """
    bounds = np.cumsum([0] + [dim ** k for k in range(1, level + 1)])
    out = np.empty_like(S, dtype=np.float64)
    for i, s in enumerate(S.astype(np.float64)):
        sq = np.array([np.sum(s[bounds[k]:bounds[k + 1]] ** 2) for k in range(level)])
        target = psi(np.sqrt(1 + sq.sum()), C, a)
        f = lambda lam: 1 + np.sum(lam ** (2 * np.arange(1, level + 1)) * sq) - target
        lam = brentq(f, 0.0, 1.0, xtol=1e-15) if f(1.0) > 0 else 1.0
        for k in range(level):
            out[i, bounds[k]:bounds[k + 1]] = lam ** (k + 1) * s[bounds[k]:bounds[k + 1]]
    return out


def robust_code(S, level, C, a):
    X = torch.tensor(S, dtype=torch.float32)
    return NadarayaWatsonSignature()._robustsignature(X, sig_level=level, C=C, a_param=a).numpy()


def large_paths(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 30, dim)).cumsum(axis=1)   # large signatures: normalisation is active


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("C, a", [(2.0, 1.0), (10.0, 0.5)])
def test_matches_definition(dim, level, C, a):
    S = ii.sig(large_paths(15, dim), level)
    expected = robust_reference(S, dim, level, C, a)
    np.testing.assert_allclose(robust_code(S, level, C, a), expected, rtol=1e-4, atol=1e-5 * np.abs(expected).max())


def test_norm_is_bounded():
    C, a = 2.0, 1.0
    R = robust_code(ii.sig(large_paths(30, 3), 3), 3, C, a)
    assert np.all(1 + (R.astype(np.float64) ** 2).sum(axis=1) <= C * (1 + 1 / a) + 1e-4)


def test_small_signatures_are_unchanged():
    """For |S|^2 <= C the normalisation is the identity (lambda = 1)."""
    rng = np.random.default_rng(0)
    S = ii.sig(rng.normal(size=(10, 30, 2)).cumsum(axis=1) * 0.01, 3)
    np.testing.assert_allclose(robust_code(S, 3, 10.0, 1.0), S, rtol=1e-5)


def test_distances_invariant_to_channel_permutation_and_rotation():
    X = large_paths(20, 3)
    Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    dist = lambda F: np.linalg.norm(F[:, None].astype(np.float64) - F[None], axis=2)
    D = dist(robust_code(ii.sig(X, 3), 3, 2.0, 1.0))
    for transformed in (X[:, :, [2, 1, 0]], X @ Q.T):
        D2 = dist(robust_code(ii.sig(transformed, 3), 3, 2.0, 1.0))
        np.testing.assert_allclose(D2, D, rtol=1e-4, atol=1e-5 * D.max())


def test_wrong_signature_length_raises():
    S = ii.sig(large_paths(5, 3), 3)
    with pytest.raises(ValueError, match="does not correspond"):
        robust_code(S, 2, 2.0, 1.0)


def test_no_normalising_factor_for_C_below_one_raises():
    S = ii.sig(large_paths(5, 2), 2)
    with pytest.raises(ValueError, match="robust_C < 1"):
        robust_code(S, 2, 0.5, 1.0)
