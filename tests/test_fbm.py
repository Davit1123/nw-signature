"""Simulation of fractional Brownian motion."""
import warnings

import numpy as np
import pytest

from nwsig import FBM, fBm_sampling, fbm, fgn


@pytest.mark.parametrize("method", ["daviesharte", "cholesky", "hosking"])
def test_shapes(method):
    for M in (1, 3):
        B, G, W = FBM(n=16, M=M, hurst=0.7, method=method).fbm()
        assert B.shape == (17, M) and G.shape == (16, M) and W.shape == (16, M)
        np.testing.assert_array_equal(B[0], 0.0)


def test_helpers():
    B, G, W = fbm(32, 0.7)
    assert B.shape == (33, 1)
    G, W = fgn(32, 0.7, M=4)
    assert G.shape == (32, 4)
    paths, increments, bm = fBm_sampling(5, 32, 0.3)
    assert paths.shape == (5, 33) and increments.shape == (5, 32)


@pytest.mark.parametrize("method", ["daviesharte", "cholesky", "hosking"])
@pytest.mark.parametrize("hurst", [0.3, 0.7])
def test_covariance(method, hurst):
    """Cov(B_s, B_t) = (s^2H + t^2H - |t - s|^2H) / 2 on [0, 1]."""
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B, _, _ = FBM(n=32, M=4000, hurst=hurst, method=method).fbm()
    t = np.linspace(0, 1, 33)
    for i, j in [(8, 32), (16, 32), (16, 24), (32, 32)]:
        expected = 0.5 * (t[i] ** (2 * hurst) + t[j] ** (2 * hurst) - abs(t[j] - t[i]) ** (2 * hurst))
        assert np.mean(B[i] * B[j]) == pytest.approx(expected, abs=0.08)
