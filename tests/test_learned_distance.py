"""Nadaraya-Watson estimators with a learned signature distance."""
import inspect
import warnings

import iisignature as ii
import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

from nwsig import (NadarayaWatsonSignature, NWLearnedDistance, NWLearnedDistanceClassifier,
                   NWLearnedDistanceRegressor, NWSignatureClassifier, NWSignatureRegressor)
from test_robust_signature import robust_reference


def scale_levels(S, dim, level, weights):
    """Multiply signature level k (dim**k entries) by sqrt(weights[k-1])."""
    sizes = [dim ** k for k in range(1, level + 1)]
    return S * np.repeat(np.sqrt(weights), sizes)


def brute_force_loo(S, y, h, task):
    """
    LOO loss by refitting the plain estimator without each sample on the
    signatures S; NadarayaWatsonSignature's bandwidth h' = 2 h^2 gives the
    weights exp(-d^2 / (2 h^2)).
    """
    losses = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        if task == "regression":
            model = NWSignatureRegressor(input_type="signatures", signature_level=3, bandwidth=2 * h ** 2)
            pred = model.fit(S[keep], y[keep]).predict(S[i:i + 1])[0]
            losses.append((pred - y[i]) ** 2)
        else:
            model = NWSignatureClassifier(input_type="signatures", signature_level=3, bandwidth=2 * h ** 2)
            proba = model.fit(S[keep], y[keep]).predict_proba(S[i:i + 1])[0]
            losses.append(-np.log(proba[list(model.classes_).index(y[i])]))
    return np.mean(losses)


@pytest.fixture
def noisy_classification_data(make_paths):
    """Classes overlap, so the LOO cross-entropy stays away from 0."""
    X = make_paths(60, seed=3)
    noise = np.random.default_rng(0).normal(scale=0.4, size=60)
    return X, np.digitize(X[:, -1, 1] + noise, [-0.3, 0.3])


# --------------------------------------------------------------------------- definition

@pytest.mark.parametrize("robust", [False, True])
def test_unit_weights_equal_plain_estimator(regression_data, classification_data, robust):
    """Without learning, the estimator is the plain one with bandwidth h' = 2 h^2."""
    h = 0.3
    kwargs = dict(signature_level=3, robust=robust, robust_C=2, robust_a=1)
    X, y = regression_data
    learned = NWLearnedDistanceRegressor(bandwidth=h, max_iter=0, **kwargs).fit(X[:60], y[:60])
    plain = NWSignatureRegressor(bandwidth=2 * h ** 2, **kwargs).fit(X[:60], y[:60])
    np.testing.assert_allclose(learned.level_weights_, 1.0)
    np.testing.assert_allclose(learned.predict(X[60:]), plain.predict(X[60:]), rtol=1e-5)

    X, y = classification_data
    learned = NWLearnedDistanceClassifier(bandwidth=h, max_iter=0, **kwargs).fit(X[:60], y[:60])
    plain = NWSignatureClassifier(bandwidth=2 * h ** 2, **kwargs).fit(X[:60], y[:60])
    np.testing.assert_allclose(learned.predict_proba(X[60:]), plain.predict_proba(X[60:]), atol=1e-5)


def test_regression_loo_loss_matches_refitting(regression_data):
    X, y = regression_data
    X, y = X[:40], y[:40]
    model = NWLearnedDistanceRegressor(signature_level=3).fit(X, y)
    h, S = model.best_bandwidth_, ii.sig(X, 3)
    assert brute_force_loo(S, y, h, "regression") == pytest.approx(model.loo_loss_init_, rel=1e-4)
    S_learned = scale_levels(S, 2, 3, model.level_weights_)
    assert brute_force_loo(S_learned, y, h, "regression") == pytest.approx(model.loo_loss_, rel=1e-4)
    # predictions use the learned distance
    np.testing.assert_allclose(model.X_train_.numpy(), S_learned, rtol=1e-5, atol=1e-6)


def test_classification_loo_loss_matches_refitting(noisy_classification_data):
    X, y = noisy_classification_data
    model = NWLearnedDistanceClassifier(signature_level=3).fit(X, y)
    h, S = model.best_bandwidth_, ii.sig(X, 3)
    assert 0.1 < model.loo_loss_ < model.loo_loss_init_
    assert brute_force_loo(S, y, h, "classification") == pytest.approx(model.loo_loss_init_, rel=1e-4)
    S_learned = scale_levels(S, 2, 3, model.level_weights_)
    assert brute_force_loo(S_learned, y, h, "classification") == pytest.approx(model.loo_loss_, rel=1e-4)


def test_robust_features_are_reweighted(regression_data):
    X, y = regression_data
    model = NWLearnedDistanceRegressor(signature_level=3, robust=True, robust_C=2, robust_a=1).fit(X, y)
    expected = scale_levels(robust_reference(ii.sig(X, 3), 2, 3, 2.0, 1.0), 2, 3, model.level_weights_)
    np.testing.assert_allclose(model.X_train_.numpy(), expected, rtol=1e-4, atol=1e-5)


def test_log_signature_levels(make_paths):
    X = make_paths(50, dim=2)
    y = X[:, -1, 1] ** 2
    model = NWLearnedDistanceRegressor(signature_level=4, log_signature=True).fit(X, y)
    # Lyndon words of length 1, 2, 3, 4 on two letters
    assert model.level_sizes_ == [2, 1, 2, 3]
    L = ii.logsig(X, ii.prepare(2, 4, "O"))
    np.testing.assert_allclose(model.X_train_.numpy(), L * np.repeat(np.sqrt(model.level_weights_), [2, 1, 2, 3]),
                               rtol=1e-5, atol=1e-6)


def principal_directions(S, r):
    """Top r principal directions of S (rows), computed with numpy."""
    return np.linalg.svd(S - S.mean(axis=0), full_matrices=False)[2][:r]


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_projection_loo_loss_matches_refitting(regression_data, noisy_classification_data, task):
    X, y = regression_data if task == "regression" else noisy_classification_data
    X, y = X[:40], y[:40]
    Model = NWLearnedDistanceRegressor if task == "regression" else NWLearnedDistanceClassifier
    model = Model(signature_level=3, distance="projection", projection_dim=3).fit(X, y)
    h, S = model.best_bandwidth_, ii.sig(X, 3)
    assert model.projection_.shape == (3, S.shape[1])
    assert model.loo_loss_ < model.loo_loss_init_
    # the initial distance is the projection on the top principal directions (signs do not matter)
    init = brute_force_loo(S @ principal_directions(S, 3).T, y, h, task)
    assert init == pytest.approx(model.loo_loss_init_, rel=1e-4)
    assert brute_force_loo(S @ model.projection_.T, y, h, task) == pytest.approx(model.loo_loss_, rel=1e-4)
    np.testing.assert_allclose(model.X_train_.numpy(), S @ model.projection_.T, rtol=1e-4, atol=1e-5)


def test_full_rank_projection_starts_from_the_unweighted_metric(regression_data):
    """With as many directions as features the initial projection is a rotation."""
    X, y = regression_data
    kwargs = dict(signature_level=2, max_iter=0)
    plain = NWLearnedDistanceRegressor(**kwargs).fit(X, y)
    rotated = NWLearnedDistanceRegressor(distance="projection", projection_dim=6, **kwargs).fit(X, y)
    assert rotated.projection_.shape == (6, 6)
    np.testing.assert_allclose(rotated.projection_ @ rotated.projection_.T, np.eye(6), atol=1e-10)
    assert rotated.loo_loss_init_ == pytest.approx(plain.loo_loss_init_, rel=1e-10)
    np.testing.assert_allclose(rotated.predict(X), plain.predict(X), rtol=1e-4)


def test_projection_dim_is_capped(make_paths):
    X = make_paths(5)
    model = NWLearnedDistanceRegressor(signature_level=2, distance="projection", projection_dim=50).fit(X, X[:, -1, 1])
    assert model.projection_.shape == (5, 6)


# --------------------------------------------------------------------------- learning

def synthetic_levels(n, task, seed=0):
    """
    Pre-computed 'signatures' of a 2-dimensional path at level 2 (2 + 4
    features): level 1 is large noise, the target depends on level 2 only.
    """
    rng = np.random.default_rng(seed)
    S = np.concatenate([10 * rng.normal(size=(n, 2)), rng.normal(size=(n, 4))], axis=1)
    target = S[:, 2] + np.sin(2 * S[:, 3])
    return S, (target if task == "regression" else (target > 0).astype(int))


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_learns_the_informative_level(task):
    S, y = synthetic_levels(300, task)
    Model = NWLearnedDistanceRegressor if task == "regression" else NWLearnedDistanceClassifier
    kwargs = dict(input_type="signatures", signature_level=2)
    plain = Model(max_iter=0, **kwargs).fit(S[:200], y[:200])
    learned = Model(**kwargs).fit(S[:200], y[:200])
    weights = learned.level_weights_
    assert weights[0] < 0.01 * weights[1]
    # at least as good as the plain estimator on level 2 alone (level 1 dropped by hand),
    # which the learned distance contains in the limit weights[0] -> 0
    S_oracle = S.copy()
    S_oracle[:, :2] = 0
    oracle = Model(max_iter=0, **kwargs).fit(S_oracle[:200], y[:200])
    assert learned.loo_loss_ < 1.01 * oracle.loo_loss_ < 0.6 * plain.loo_loss_
    assert learned.score(S[200:], y[200:]) > plain.score(S[200:], y[200:]) + 0.2


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_projection_finds_the_informative_direction(task):
    """
    Six features: two of large noise (which the initial principal directions
    pick), the target depends on z_2 - z_4 only.
    """
    rng = np.random.default_rng(0)
    S = rng.normal(size=(300, 6)) * np.array([10, 10, 1, 1, 1, 1])
    target = S[:, 2] - S[:, 4]
    y = target if task == "regression" else (target > 0).astype(int)
    Model = NWLearnedDistanceRegressor if task == "regression" else NWLearnedDistanceClassifier
    kwargs = dict(input_type="signatures", signature_level=2, distance="projection", projection_dim=2)
    plain = Model(max_iter=0, **kwargs).fit(S[:200], y[:200])
    learned = Model(**kwargs).fit(S[:200], y[:200])
    u = np.array([0, 0, 1, 0, -1, 0]) / np.sqrt(2)
    B = learned.projection_
    # u lies (almost) in the row space of B
    in_row_space = np.linalg.norm(np.linalg.pinv(B) @ (B @ u))
    assert in_row_space > 0.95
    assert learned.loo_loss_ < 0.05 * plain.loo_loss_
    assert learned.score(S[200:], y[200:]) > plain.score(S[200:], y[200:]) + 0.5


@pytest.mark.parametrize("distance", ["levels", "projection"])
def test_learning_does_not_increase_loo_loss(regression_data, classification_data, distance):
    for Model, (X, y) in [(NWLearnedDistanceRegressor, regression_data),
                          (NWLearnedDistanceClassifier, classification_data)]:
        model = Model(signature_level=3, distance=distance, projection_dim=4).fit(X, y)
        assert model.loo_loss_ <= model.loo_loss_init_
        assert model.n_iter_ >= 1


def test_selects_representation_by_loo(regression_data):
    X, y = regression_data
    model = NWLearnedDistanceRegressor(signature_level_grid=[1, 2, 3], robust=True,
                                       robust_C_grid=[1, 5], robust_a=1).fit(X, y)
    assert model.best_signature_level_ in (1, 2, 3) and model.best_robust_C_ in (1, 5)
    assert len(model.level_weights_) == model.best_signature_level_
    for level in (1, 2, 3):
        for C in (1, 5):
            other = NWLearnedDistanceRegressor(signature_level=level, robust=True, robust_C=C, robust_a=1,
                                               max_iter=0).fit(X, y)
            assert model.loo_loss_init_ <= other.loo_loss_init_ + 1e-12


def test_loo_subsample(regression_data):
    X, y = regression_data
    fit = lambda: NWLearnedDistanceRegressor(signature_level=3, loo_max_samples=25).fit(X[:60], y[:60])
    model = fit()
    assert model.loo_loss_ <= model.loo_loss_init_
    np.testing.assert_array_equal(model.predict(X[60:]), fit().predict(X[60:]))


@pytest.mark.parametrize("distance", ["levels", "projection"])
def test_class_with_single_sample(make_paths, distance):
    X = make_paths(30)
    y = np.array([0] * 15 + [1] * 14 + [2])
    model = NWLearnedDistanceClassifier(signature_level=2, distance=distance, projection_dim=3).fit(X, y)
    assert np.isfinite(model.loo_loss_)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(model.predict(X), model.classes_[proba.argmax(axis=1)])


# --------------------------------------------------------------------------- scikit-learn integration

def test_get_params_and_clone(regression_data):
    own = set(inspect.signature(NWLearnedDistance.__init__).parameters) - {"self", "kwargs"}
    base = set(inspect.signature(NadarayaWatsonSignature.__init__).parameters) - {"self"}
    model = NWLearnedDistanceRegressor(distance="projection", projection_dim=3, max_iter=7, learning_rate=0.3,
                                       loo_max_samples=50, signature_level=3)
    assert set(model.get_params()) == own | base
    copy = clone(model)
    assert (copy.distance, copy.projection_dim, copy.max_iter, copy.learning_rate, copy.loo_max_samples,
            copy.signature_level, copy.task_type) == ("projection", 3, 7, 0.3, 50, 3, "regression")
    X, y = regression_data
    before = model.get_params()
    model.fit(X, y)
    assert model.get_params() == before


def test_sklearn_tools(classification_data):
    assert is_regressor(NWLearnedDistanceRegressor()) and is_classifier(NWLearnedDistanceClassifier())
    X, y = classification_data
    search = GridSearchCV(NWLearnedDistanceClassifier(), {"signature_level": [1, 2], "max_iter": [0, 20]}, cv=3)
    assert search.fit(X, y).score(X, y) > 0.5
    with pytest.raises(NotFittedError):
        NWLearnedDistanceClassifier().predict(X)


@pytest.mark.parametrize("kwargs", [
    dict(distance="mahalanobis"),
    dict(distance="projection", projection_dim=0),
    dict(metric_choice="l2"),
    dict(initial_condition=True),
    dict(max_iter=-1),
    dict(max_iter=2.5),
    dict(learning_rate=0),
    dict(loo_max_samples=1),
    dict(bandwidth=0),
    dict(robust=True, robust_C=0.5),
])
def test_invalid_parameters_raise_at_fit(regression_data, kwargs):
    X, y = regression_data
    model = NWLearnedDistanceRegressor(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            model.fit(X, y)
