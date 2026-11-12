"""Windows chosen by greedy binary splitting."""
import iisignature as ii
import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline

from nwsig import NWSignatureClassifier, NWSignatureRegressor, WindowTree


def window_data(n=120, n_timesteps=65, seed=0):
    """Brownian paths whose class is the sign of the increment of channel 1 on [32, 48]."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_timesteps)
    W = np.concatenate([np.zeros((n, 1, 2)), rng.normal(size=(n, n_timesteps - 1, 2)).cumsum(axis=1) / 8], axis=1)
    X = np.concatenate([np.broadcast_to(t[None, :, None], (n, n_timesteps, 1)), W], axis=2)
    y = (X[:, 48, 1] > X[:, 32, 1]).astype(int)
    return X, y


def brute_force_loo(F, y, h, task, criterion="brier"):
    """LOO loss by refitting the estimator on the features F without each sample (weights exp(-d^2 / h))."""
    losses = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        if task == "regression":
            model = NWSignatureRegressor(input_type="signatures", bandwidth=h).fit(F[keep], y[keep])
            losses.append((model.predict(F[i:i + 1])[0] - y[i]) ** 2)
        else:
            model = NWSignatureClassifier(input_type="signatures", bandwidth=h).fit(F[keep], y[keep])
            proba = dict(zip(model.classes_, model.predict_proba(F[i:i + 1])[0]))
            if criterion == "brier":
                losses.append(sum((proba.get(c, 0.0) - (c == y[i])) ** 2 for c in np.unique(y)))
            else:
                losses.append(-np.log(proba[y[i]]))
    return np.mean(losses)


# --------------------------------------------------------------------- pieces

@pytest.mark.parametrize("n_timesteps, n_segments", [(65, 8), (100, 32), (10, 32), (2, 4)])
def test_grid(n_timesteps, n_segments):
    grid = WindowTree(n_segments=n_segments)._grid(n_timesteps)
    assert grid[0] == 0 and grid[-1] == n_timesteps - 1
    assert np.all(np.diff(grid) > 0)
    assert len(grid) == min(n_segments, n_timesteps - 1) + 1


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_window_signatures_from_segments(make_paths, gamma):
    # Chen's identity: the product of the segment signatures is the signature of the window
    X = make_paths(5, n_timesteps=41, dim=3)
    tree = WindowTree(level=3, n_segments=8)
    tree.dim_, tree.grid_ = 3, tree._grid(41)
    segments = tree._segment_signatures(X)
    for i, j in [(0, 8), (0, 3), (2, 7), (5, 6)]:
        a, b = tree.grid_[i], tree.grid_[j]
        lam = (40 / (b - a)) ** gamma
        expected = ii.sig(np.ascontiguousarray(lam * X[:, a:b + 1]), 3)
        np.testing.assert_allclose(tree._window_signatures(segments, i, j, gamma), expected,
                                   rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("task, criterion", [("classification", "brier"), ("classification", "cross_entropy"),
                                             ("regression", "brier")])
def test_loo_loss_matches_refitting(make_paths, task, criterion):
    X = make_paths(40, dim=3)
    # three classes, so that the Brier score is not a multiple of the two-class one
    y = np.digitize(X[:, -1, 1], [-0.3, 0.3]) if task == "classification" else X[:, -1, 1] + X[:, 20, 2]
    F = ii.sig(X, 2)
    tree = WindowTree(task=task, criterion=criterion)
    tree._rows = np.arange(len(y))
    loss = tree._make_loo_loss(y)
    D2 = torch.cdist(torch.as_tensor(F), torch.as_tensor(F)) ** 2
    for h in (0.01, 0.1, 1.0):
        np.testing.assert_allclose(loss(D2, h), brute_force_loo(F, y, h, task, criterion), rtol=1e-4)


def test_loo_loss_on_a_subset_of_rows(make_paths):
    X = make_paths(30, dim=3)
    y = (X[:, -1, 1] > 0).astype(int)
    F = torch.as_tensor(ii.sig(X, 2))
    D2 = torch.cdist(F, F) ** 2
    full, part = WindowTree(), WindowTree()
    full._rows, part._rows = np.arange(30), np.array([1, 4, 9, 20])
    per_sample = []
    for i in part._rows:
        single = WindowTree()
        single._rows = np.array([i])
        per_sample.append(single._make_loo_loss(y)(D2[[i]], 0.1))
    np.testing.assert_allclose(part._make_loo_loss(y)(D2[part._rows], 0.1), np.mean(per_sample), rtol=1e-6)   # float32


# ---------------------------------------------------------------------- fit

def test_no_split_is_the_signature_metric(make_paths):
    X = make_paths(20, dim=3)
    y = np.arange(20) % 2
    tree = WindowTree(level=2, max_splits=0).fit(X, y)
    assert tree.windows_ == [(0, 40)] and list(tree.weights_) == [1.0] and tree.depths_ == [0]
    assert tree.loo_path_ == [tree.loo_loss_init_] and tree.loo_loss_ == tree.loo_loss_init_
    np.testing.assert_allclose(tree.transform(X), ii.sig(X, 2) / np.sqrt(tree.medians_[0]), rtol=1e-10)


def test_transform_distance_is_the_tree_distance():
    X, y = window_data(60)
    tree = WindowTree(level=2, n_segments=8, max_splits=3, min_gain=0.0).fit(X, y)
    assert len(tree.windows_) > 1
    F = tree.transform(X)
    lam = lambda a, b: (64 / (b - a)) ** tree.gamma_
    S = [ii.sig(np.ascontiguousarray(lam(a, b) * X[:, a:b + 1]), 2) for a, b in tree.windows_]
    for i, k in [(0, 1), (3, 17), (25, 59)]:
        expected = sum(w / m * np.sum((s[i] - s[k]) ** 2) for s, w, m in zip(S, tree.weights_, tree.medians_))
        np.testing.assert_allclose(np.sum((F[i] - F[k]) ** 2), expected, rtol=1e-9)


def test_tree_finds_the_informative_window():
    X, y = window_data()
    tree = WindowTree(level=2, n_segments=8).fit(X, y)
    assert (32, 48) in tree.windows_
    assert tree.loo_loss_ < 0.8 * tree.loo_loss_init_
    assert all(b < a for a, b in zip(tree.loo_path_[:-1], tree.loo_path_[1:]))   # every split helps
    assert all(d <= tree.max_depth for d in tree.depths_)
    for (a, b), d in zip(tree.windows_[1:], tree.depths_[1:]):
        assert b - a >= tree.min_steps and d >= 1


def test_windows_form_a_binary_tree():
    X, y = window_data()
    tree = WindowTree(level=2, n_segments=8, min_gain=0.0, max_splits=6).fit(X, y)
    windows = tree.windows_
    for k in range(1, len(windows), 2):
        (a, s), (s2, b) = windows[k], windows[k + 1]
        assert s == s2 and (a, b) in windows[:k]            # the two halves of an earlier window
        assert tree.weights_[k] == tree.weights_[k + 1]


@pytest.mark.parametrize("criterion", ["brier", "cross_entropy"])
def test_random_labels_barely_improve(criterion):
    X, y = window_data()
    real = WindowTree(level=2, n_segments=8, criterion=criterion).fit(X, y)
    shuffled = WindowTree(level=2, n_segments=8, criterion=criterion).fit(X, np.random.default_rng(1).permutation(y))
    gain = lambda t: 1 - t.loo_loss_ / t.loo_loss_init_
    assert gain(shuffled) < 0.5 * gain(real)


def test_level_one_uses_gamma_zero(make_paths):
    X = make_paths(20, dim=3)
    tree = WindowTree(level=1, max_splits=1).fit(X, np.arange(20) % 2)
    assert tree.gamma_ == 0.0 and list(tree.gamma_losses_) == [0.0]


def test_regression():
    X, _ = window_data()
    y = X[:, 48, 1] - X[:, 32, 1]
    tree = WindowTree(level=2, task="regression", n_segments=8).fit(X, y)
    assert tree.loo_loss_ < 0.8 * tree.loo_loss_init_


def test_loo_subsample(make_paths):
    X, y = window_data(80)
    tree = WindowTree(level=2, n_segments=8, max_splits=2, loo_max_samples=30).fit(X, y)
    assert len(tree._rows) == 30 and tree.transform(X).shape[0] == 80


def test_pipeline_with_the_estimator():
    X, y = window_data()
    model = make_pipeline(WindowTree(level=2, n_segments=8),
                          NWSignatureClassifier(input_type="signatures", signature_level=2))
    plain = NWSignatureClassifier(signature_level=2).fit(X[:80], y[:80])
    model.fit(X[:80], y[:80])
    assert model.score(X[80:], y[80:]) > plain.score(X[80:], y[80:])
    clone(model)


# ------------------------------------------------------------------- errors

def test_not_fitted(make_paths):
    with pytest.raises(NotFittedError):
        WindowTree().transform(make_paths(3))


def test_transform_needs_the_training_shape(make_paths):
    tree = WindowTree(max_splits=0).fit(make_paths(10), np.arange(10) % 2)
    with pytest.raises(ValueError):
        tree.transform(make_paths(3, n_timesteps=30))


@pytest.mark.parametrize("kwargs", [dict(level=0), dict(task="ranking"), dict(criterion="gini"),
                                    dict(n_segments=0), dict(max_depth=-1),
                                    dict(min_gain=1.0), dict(weight_grid=(0.0, 1.0)), dict(gamma_grid=()),
                                    dict(loo_max_samples=1)])
def test_invalid_parameters(make_paths, kwargs):
    with pytest.raises(ValueError):
        WindowTree(**kwargs).fit(make_paths(10), np.arange(10) % 2)


def test_invalid_data(make_paths):
    with pytest.raises(ValueError):
        WindowTree().fit(np.zeros((5, 10)), np.zeros(5))
    with pytest.raises(ValueError):
        WindowTree().fit(make_paths(10), np.zeros(9))
