"""Nadaraya-Watson classifier with a learned multi-scale signature distance."""
import warnings

import iisignature as ii
import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from nwsig import NWLearnedWindowClassifier, NWSignatureClassifier, multiscale_windows, window_increments_and_areas
from nwsig.learned_window import _contract
from test_window_tree import window_data


def levy_area(S2, c):
    """(S_ij - S_ji) / 2 for i < j from a flat level-2 signature."""
    S2 = S2.reshape(len(S2), c, c)
    i, j = np.triu_indices(c, k=1)
    return 0.5 * (S2[:, i, j] - S2[:, j, i])


# ----------------------------------------------------------------- windows

@pytest.mark.parametrize("n_timesteps, n_scales, shifted, n_windows", [
    (65, 4, True, 15 + 11), (65, 4, False, 15), (65, 1, True, 1), (8, 4, True, 4), (2, 3, True, 1), (3, 3, True, 1)])
def test_multiscale_windows(n_timesteps, n_scales, shifted, n_windows):
    grid, windows = multiscale_windows(n_timesteps, n_scales, shifted)
    assert len(windows) == n_windows
    assert grid[0] == 0 and grid[-1] == n_timesteps - 1 and np.all(np.diff(grid) >= 1)
    assert windows[0][:2] == (0, len(grid) - 1)                     # the whole path first
    for i, j, scale in windows:
        assert (j - i) * 2 ** scale == len(grid) - 1                # width 1/2**scale of the path


def test_scales_are_halved():
    grid, windows = multiscale_windows(65, 3)
    assert grid.tolist() == list(range(0, 65, 8))
    assert [w for w in windows if w[2] == 1] == [(0, 4, 1), (4, 8, 1), (2, 6, 1)]


def test_increments_and_areas_are_the_log_signature(make_paths):
    X = make_paths(6, n_timesteps=33, dim=4)
    grid, windows = multiscale_windows(33, 3)
    starts, ends = torch.tensor([w[0] for w in windows]), torch.tensor([w[1] for w in windows])
    inc, area = window_increments_and_areas(torch.as_tensor(X), torch.as_tensor(grid), starts, ends)
    for k, (i, j, _) in enumerate(windows):
        S = ii.sig(np.ascontiguousarray(X[:, grid[i]:grid[j] + 1]), 2)
        np.testing.assert_allclose(inc[:, k].numpy(), S[:, :4], atol=1e-12)
        np.testing.assert_allclose(area[:, k].numpy(), levy_area(S[:, 4:], 4), atol=1e-12)


@pytest.mark.parametrize("k", [1, 3, 4])
def test_contraction_is_the_signature_of_the_mapped_path(make_paths, k):
    # S^k(M x) = M^{(x)k} S^k(x)
    X = make_paths(5, dim=4)
    M = np.random.default_rng(0).normal(size=(3, 4))
    bounds = np.cumsum([0] + [4 ** j for j in range(1, k + 1)])
    S = torch.as_tensor(ii.sig(X, k)[:, bounds[k - 1]:bounds[k]])
    expected = ii.sig(np.ascontiguousarray(X @ M.T), k)[:, -3 ** k:]
    np.testing.assert_allclose(_contract(S, torch.as_tensor(M), k).numpy(), expected, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------- fit

@pytest.fixture(scope="module")
def data():
    X, y = window_data(300)
    return X[:200], y[:200], X[200:], y[200:]


@pytest.fixture(scope="module")
def fitted(data):
    Xtr, ytr, _, _ = data
    return NWLearnedWindowClassifier(time_channel=0, level=3).fit(Xtr, ytr)


def test_learns_the_informative_window(data, fitted):
    _, _, Xte, yte = data
    plain = NWSignatureClassifier(signature_level=3).fit(data[0], data[1])
    assert fitted.score(Xte, yte) > plain.score(Xte, yte) + 0.1
    assert fitted.history_[-1]["train"] < fitted.history_[0]["train"]


def test_distance_is_the_weighted_sum_of_blocks(data, fitted):
    X = data[2][:6]
    F = fitted.transform(X)
    M = fitted.projection_
    Z = X @ M.T
    c = M.shape[0]
    expected = np.zeros((6, 6))
    for key, w in fitted.block_weights_.items():
        if key[0] == "whole path":
            k = key[1]
            S = ii.sig(np.ascontiguousarray(Z), k)[:, -c ** k:]
        else:
            scale, kind = key
            parts = []
            for a, b, s in fitted.windows_:
                if s != scale:
                    continue
                S2 = ii.sig(np.ascontiguousarray(Z[:, a:b + 1]), 2)
                parts.append(S2[:, :c] if kind == "increments" else levy_area(S2[:, c:], c))
            S = np.concatenate(parts, axis=1)
        expected += w * ((S[:, None] - S[None]) ** 2).sum(axis=2)
    np.testing.assert_allclose(((F[:, None] - F[None]) ** 2).sum(axis=2), expected, rtol=2e-3, atol=1e-4)


def test_bandwidth_selection(data, fitted):
    assert 0 < fitted.cv_accuracy_ <= 1
    Xtr, ytr, Xte, yte = data
    brier = NWLearnedWindowClassifier(time_channel=0, level=3, bandwidth_selection="validation_brier").fit(Xtr, ytr)
    assert np.isnan(brier.cv_accuracy_)
    # same learned shape of the distance, only its overall scale differs
    ratio = np.array(list(fitted.block_weights_.values())) / np.array(list(brier.block_weights_.values()))
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-4)
    np.testing.assert_allclose(fitted.projection_, brier.projection_)


def test_time_channel_is_passed_through(fitted):
    M = fitted.projection_
    assert M[0, 0] == 1 and np.all(M[0, 1:] == 0) and np.all(M[1:, 0] == 0)


def test_proba(data, fitted):
    P = fitted.predict_proba(data[2])
    assert P.shape == (100, 2)
    np.testing.assert_allclose(P.sum(axis=1), 1, rtol=1e-5)
    np.testing.assert_array_equal(fitted.predict(data[2]), fitted.classes_[P.argmax(axis=1)])


def test_weights_only_keeps_the_initial_map(data):
    Xtr, ytr, _, _ = data
    model = NWLearnedWindowClassifier(time_channel=0, level=2, learn_projection=False, max_iter=20).fit(Xtr, ytr)
    np.testing.assert_array_equal(model.projection_, np.eye(3))       # 2 channels + time: identity


def test_projection_to_fewer_channels(make_paths):
    X = make_paths(60, dim=6)
    y = (X[:, -1, 1] + X[:, -1, 2] > 0).astype(int)
    model = NWLearnedWindowClassifier(projection_dim=2, time_channel=0, level=3, max_iter=20).fit(X, y)
    assert model.projection_.shape == (3, 6)
    assert set(model.block_weights_) >= {("whole path", 3)}
    assert model.transform(X).shape[0] == 60


def test_string_labels_and_minibatches(data):
    Xtr, ytr, Xte, _ = data
    labels = np.array(["down", "up"])
    model = NWLearnedWindowClassifier(time_channel=0, level=2, batch_size=64, max_iter=30).fit(Xtr, labels[ytr])
    assert set(model.predict(Xte)) <= {"down", "up"} and list(model.classes_) == ["down", "up"]


def test_deterministic(data):
    Xtr, ytr, Xte, _ = data
    kwargs = dict(time_channel=0, level=2, max_iter=30)
    P1 = NWLearnedWindowClassifier(**kwargs).fit(Xtr, ytr).predict_proba(Xte)
    P2 = NWLearnedWindowClassifier(**kwargs).fit(Xtr, ytr).predict_proba(Xte)
    np.testing.assert_allclose(P1, P2, rtol=1e-5)


def test_auto_level(make_paths):
    X = make_paths(30, dim=3)
    model = NWLearnedWindowClassifier(max_iter=0).fit(X, np.arange(30) % 2)
    assert model.level_ == 5 and model.n_iter_ == 0


def test_no_warnings(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NWLearnedWindowClassifier(time_channel=0, level=2, max_iter=20).fit(data[0], data[1])


def test_clone(fitted):
    assert clone(fitted).get_params() == fitted.get_params()


# ------------------------------------------------------------------ errors

def test_not_fitted(make_paths):
    with pytest.raises(NotFittedError):
        NWLearnedWindowClassifier().predict(make_paths(3))


def test_predict_needs_the_training_shape(fitted, make_paths):
    with pytest.raises(ValueError):
        fitted.predict(make_paths(3, n_timesteps=30, dim=3))


@pytest.mark.parametrize("kwargs", [dict(level=1), dict(n_scales=0), dict(batch_size=1), dict(validation_fraction=0.0),
                                    dict(learning_rate=0.0), dict(time_channel=5),
                                    dict(bandwidth_selection="median"), dict(cv_folds=1)])
def test_invalid_parameters(make_paths, kwargs):
    with pytest.raises(ValueError):
        NWLearnedWindowClassifier(max_iter=1, **kwargs).fit(make_paths(20), np.arange(20) % 2)
