"""p-variation against a brute-force dynamic programme."""
import numpy as np
import pytest

from nwsig.p_var_vectorized import p_variation, pvar_backbone_matrix_numba


def brute_force(x, p):
    """max over partitions of sum |x_{t_(i+1)} - x_{t_i}|^p, O(T^2)."""
    best = np.zeros(len(x))
    for j in range(1, len(x)):
        best[j] = max(best[m] + np.linalg.norm(x[j] - x[m]) ** p for m in range(j))
    return best[-1]


@pytest.mark.parametrize("n_timesteps", [2, 3, 8, 33, 100])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 4.0])
def test_matches_brute_force(n_timesteps, dim, p):
    x = np.random.default_rng(n_timesteps * dim).normal(size=(n_timesteps, dim)).cumsum(axis=0)
    result = p_variation(x, p)
    assert result.value == pytest.approx(brute_force(x, p), rel=1e-12)
    # the returned partition attains the value
    pts = result.points
    assert pts[0] == 0 and pts[-1] == n_timesteps - 1 and np.all(np.diff(pts) > 0)
    total = sum(np.linalg.norm(x[b] - x[a]) ** p for a, b in zip(pts[:-1], pts[1:]))
    assert total == pytest.approx(result.value, rel=1e-12)


def test_matrix_is_p_variation_of_difference_paths():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(4, 50, 2)).cumsum(axis=1), rng.normal(size=(3, 50, 2)).cumsum(axis=1)
    M = pvar_backbone_matrix_numba(A, B, 2.0)
    expected = np.array([[p_variation(a - b, 2.0).value for b in B] for a in A])
    np.testing.assert_allclose(M, expected, rtol=1e-12)


def test_edge_cases():
    assert p_variation(np.zeros((1, 2))).value == 0.0
    assert p_variation(np.zeros((0, 2))).value == -np.inf
    x = np.random.default_rng(0).normal(size=40).cumsum()
    assert p_variation(x, 2).value == p_variation(x[:, None], 2).value
    with pytest.raises(ValueError):
        p_variation(x, 0.5)
