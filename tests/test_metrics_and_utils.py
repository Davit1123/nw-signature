"""Path distances in nwsig.metrics and the helpers in nwsig.utils."""
import iisignature as ii
import numpy as np
import pytest

from nwsig import augment_with_time, suggest_truncation_level, trunc_sig_metric
from nwsig.metrics import full_sig_metric


def test_trunc_sig_metric(make_paths):
    x, y = make_paths(2)
    assert trunc_sig_metric(x, y, 3) == pytest.approx(np.linalg.norm(ii.sig(x, 3) - ii.sig(y, 3)))


def test_full_sig_metric(make_paths):
    pytest.importorskip("sigkernel")
    x, y = make_paths(2)
    assert full_sig_metric(x, x) == pytest.approx(0.0, abs=1e-6)
    assert full_sig_metric(x, y) == pytest.approx(full_sig_metric(y, x))
    # with the linear static kernel this is the distance of the untruncated signatures
    assert full_sig_metric(x, y, dyadic_order=3) == pytest.approx(np.linalg.norm(ii.sig(x, 10) - ii.sig(y, 10)), rel=1e-2)


def test_augment_with_time():
    data = np.random.default_rng(0).normal(size=(4, 3, 10))   # aeon layout (n, dim, length)
    out = augment_with_time(data)
    assert out.shape == (4, 10, 4)
    np.testing.assert_allclose(out[0, :, 0], np.linspace(0, 1, 10))
    np.testing.assert_array_equal(out[:, :, 1:], data.transpose(0, 2, 1))


@pytest.mark.parametrize("dim", [0, 1, 2, 5])
@pytest.mark.parametrize("budget", [1e3, 1e5, 5e6])
def test_suggest_truncation_level(dim, budget):
    n_obs, d = 100, dim + 1
    K = suggest_truncation_level(n_obs, 50, dim, budget)
    total = lambda k: n_obs * sum(d ** j for j in range(1, k + 1))
    assert 1 <= K <= 10
    if K > 1:
        assert total(K) <= budget
    if K < 10:
        assert total(K + 1) > budget
