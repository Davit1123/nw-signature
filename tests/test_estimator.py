"""Behaviour of NWSignatureRegressor and NWSignatureClassifier."""
import inspect
import warnings

import iisignature as ii
import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from nwsig import NadarayaWatsonSignature, NWSignatureClassifier, NWSignatureRegressor

PATH_METRICS = ["supremum_metric", "l1", "l2", "lp", "var1", "pvar"]


# --------------------------------------------------------------------------- basic API

def test_regressor_fit_predict_score(regression_data):
    X, y = regression_data
    model = NWSignatureRegressor(signature_level=2).fit(X[:60], y[:60])
    pred = model.predict(X[60:])
    assert pred.shape == (20,)
    assert np.all(np.isfinite(pred))
    assert model.score(X[60:], y[60:]) > 0.5


def test_classifier_with_string_labels(classification_data):
    X, y = classification_data
    model = NWSignatureClassifier(signature_level=2).fit(X[:60], y[:60])
    assert list(model.classes_) == ["down", "flat", "up"]
    proba = model.predict_proba(X[60:])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(model.predict(X[60:]), model.classes_[proba.argmax(axis=1)])


@pytest.mark.parametrize("metric", ["signature_metric"] + PATH_METRICS)
def test_every_metric_gives_finite_predictions(regression_data, metric):
    X, y = regression_data
    model = NWSignatureRegressor(metric_choice=metric, bandwidth_grid=[1.0, 0.1]).fit(X[:60], y[:60])
    assert np.all(np.isfinite(model.predict(X[60:])))


def test_verbose_is_off_by_default(regression_data, capsys):
    X, y = regression_data
    NWSignatureRegressor(signature_level=2).fit(X, y)
    assert capsys.readouterr().out == ""
    NWSignatureRegressor(signature_level=2, verbose=True).fit(X, y)
    assert "Selected" in capsys.readouterr().out


# --------------------------------------------------------------------------- kernel weights

@pytest.mark.parametrize("bandwidth", [1e-6, 1e-3, 1.0, 100.0])
def test_predictions_are_weighted_averages_of_targets(regression_data, bandwidth):
    """A Nadaraya-Watson prediction always lies between the smallest and largest target."""
    X, y = regression_data
    y = y + 5.0
    pred = NWSignatureRegressor(signature_level=3, bandwidth=bandwidth).fit(X[:60], y[:60]).predict(X[60:])
    assert np.all(pred >= y[:60].min() - 1e-4) and np.all(pred <= y[:60].max() + 1e-4)


def test_small_bandwidth_gives_nearest_neighbour(regression_data):
    X, y = regression_data
    S_train, S_test = ii.sig(X[:60], 3), ii.sig(X[60:], 3)
    nearest = np.linalg.norm(S_test[:, None] - S_train[None], axis=2).argmin(axis=1)
    pred = NWSignatureRegressor(signature_level=3, bandwidth=1e-8).fit(X[:60], y[:60]).predict(X[60:])
    np.testing.assert_allclose(pred, y[:60][nearest], rtol=1e-5)


@pytest.mark.parametrize("kwargs", [
    dict(signature_level=2),
    dict(signature_level=3, robust=True, robust_C=2, robust_a=1),
    dict(metric_choice="l2"),
])
def test_predict_proba_is_consistent_with_predict(classification_data, kwargs):
    X, y = classification_data
    model = NWSignatureClassifier(bandwidth=0.5, **kwargs).fit(X[:60], y[:60])
    proba = model.predict_proba(X[60:])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(model.predict(X[60:]), model.classes_[proba.argmax(axis=1)])


def test_robust_model_equals_plain_model_on_robust_signatures(classification_data):
    """
    A robust model must behave exactly like a plain model fitted on robust
    signatures computed independently from the definition (fit, predict and
    predict_proba all have to apply the same transformation).
    """
    from test_robust_signature import robust_reference

    X, y = classification_data
    R = robust_reference(ii.sig(X, 3), 2, 3, 2.0, 1.0)
    robust = NWSignatureClassifier(signature_level=3, robust=True, robust_C=2, robust_a=1, bandwidth=0.05).fit(X[:60], y[:60])
    plain = NWSignatureClassifier(input_type="signatures", signature_level=3, bandwidth=0.05).fit(R[:60], y[:60])
    np.testing.assert_allclose(robust.predict_proba(X[60:]), plain.predict_proba(R[60:]), atol=1e-4)
    np.testing.assert_array_equal(robust.predict(X[60:]), plain.predict(R[60:]))

    y_reg = X[:, -1, 1] + X[:, :, 1].max(axis=1)
    robust = NWSignatureRegressor(signature_level=3, robust=True, robust_C=2, robust_a=1, bandwidth=0.05).fit(X[:60], y_reg[:60])
    plain = NWSignatureRegressor(input_type="signatures", signature_level=3, bandwidth=0.05).fit(R[:60], y_reg[:60])
    np.testing.assert_allclose(robust.predict(X[60:]), plain.predict(R[60:]), rtol=1e-4, atol=1e-5)


def test_predict_proba_matches_definition_for_l2_metric(classification_data):
    X, y = classification_data
    h = 0.05
    model = NWSignatureClassifier(metric_choice="l2", bandwidth=h).fit(X[:60], y[:60])
    D = np.sqrt((np.linalg.norm(X[60:, None] - X[None, :60], axis=3) ** 2).mean(axis=2))
    logits = -D ** 2 / h
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    expected = np.stack([w[:, y[:60] == c].sum(axis=1) for c in model.classes_], axis=1)
    np.testing.assert_allclose(model.predict_proba(X[60:]), expected, atol=1e-5)


# --------------------------------------------------------------------------- cross-validation

@pytest.mark.parametrize("metric", ["signature_metric", "l2"])
def test_cv_selects_bandwidth_with_best_mean_fold_score(regression_data, metric):
    """Compare with an independent evaluation: fixed-bandwidth fits on the same folds."""
    X, y = regression_data
    grid = [1.0, 0.3, 0.1, 0.03, 0.01]
    model = NWSignatureRegressor(metric_choice=metric, bandwidth_grid=grid, cv_folds=5, random_state=0).fit(X, y)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(X))
    mean_mse = []
    for h in grid:
        fold_mse = [mean_squared_error(y[val], NWSignatureRegressor(metric_choice=metric, bandwidth=h)
                                       .fit(X[train], y[train]).predict(X[val]))
                    for train, val in folds]
        mean_mse.append(np.mean(fold_mse))
    assert model.best_bandwidth_ == grid[int(np.argmin(mean_mse))]


def test_fixed_hyperparameters_are_kept(regression_data):
    X, y = regression_data
    # C=3 is not in the default grid; a is selected by cross-validation
    model = NWSignatureRegressor(signature_level=2, robust=True, robust_C=3).fit(X, y)
    assert model.best_robust_C_ == 3
    model = NWSignatureRegressor(signature_level=2, bandwidth=0.1, robust=True).fit(X, y)
    assert model.best_bandwidth_ == 0.1 and model.best_robust_C_ is not None
    model = NWSignatureRegressor(signature_level=2, bandwidth=0.1, robust=True, robust_C=3, robust_a=0.7).fit(X, y)
    assert (model.best_bandwidth_, model.best_robust_C_, model.best_robust_a_) == (0.1, 3, 0.7)


def test_signature_level_grid_is_searched(regression_data):
    X, y = regression_data
    model = NWSignatureRegressor(signature_level=1, signature_level_grid=[1, 2, 3], bandwidth=0.1).fit(X, y)
    assert model.best_signature_level_ in (1, 2, 3)
    assert model.X_train_.shape[1] == ii.siglength(2, model.best_signature_level_)


def test_classes_smaller_than_cv_folds(make_paths):
    """Every class smaller than cv_folds (e.g. StandWalkJump: 3 classes x 4 samples, 5 folds)."""
    X = make_paths(12)
    y = np.repeat(["jumping", "standing", "walking"], 4)
    with pytest.warns(UserWarning, match="cannot be stratified"):
        model = NWSignatureClassifier(bandwidth_grid=[1.0, 0.1], cv_folds=5).fit(X, y)
    assert set(model.predict(X)) <= set(y)


def test_cv_is_reproducible(regression_data):
    X, y = regression_data
    fit = lambda: NWSignatureRegressor(bandwidth_grid=np.geomspace(1e-3, 1, 10), random_state=3).fit(X, y)
    assert fit().best_bandwidth_ == fit().best_bandwidth_


# --------------------------------------------------------------------------- scikit-learn integration

def test_get_params_covers_every_constructor_argument():
    arguments = set(inspect.signature(NadarayaWatsonSignature.__init__).parameters) - {"self"}
    assert set(NWSignatureRegressor().get_params()) == arguments
    assert set(NWSignatureClassifier().get_params()) == arguments


def test_fit_does_not_modify_parameters_and_clone_keeps_them(regression_data):
    X, y = regression_data
    model = NWSignatureRegressor(metric_choice="pvar", p_param=4, initial_condition=True, robust=False, verbose=False)
    before = model.get_params()
    model.fit(X, y, X0=X[:, 0, :])
    assert model.get_params() == before
    copy = clone(model)
    assert (copy.metric_choice, copy.p_param, copy.initial_condition) == ("pvar", 4, True)


def test_cross_val_score_uses_the_configured_metric(regression_data):
    """cross_val_score clones the estimator; the clone must keep metric_choice."""
    X, y = regression_data
    scores = cross_val_score(NWSignatureRegressor(metric_choice="l2", bandwidth=0.1), X, y, cv=KFold(3))
    manual = [NWSignatureRegressor(metric_choice="l2", bandwidth=0.1).fit(X[train], y[train]).score(X[val], y[val])
              for train, val in KFold(3).split(X)]
    np.testing.assert_allclose(scores, manual)


def test_sklearn_tools(classification_data, regression_data):
    assert is_regressor(NWSignatureRegressor()) and is_classifier(NWSignatureClassifier())
    X, y = classification_data
    search = GridSearchCV(NWSignatureClassifier(), {"bandwidth": [1.0, 0.1], "signature_level": [1, 2]}, cv=3)
    assert search.fit(X, y).score(X, y) > 0.5
    X, y = regression_data
    scores = cross_val_score(NWSignatureRegressor(metric_choice="l2", bandwidth=0.1), X, y, cv=3)
    assert scores.shape == (3,)
    with pytest.raises(NotFittedError):
        NWSignatureRegressor().predict(X)
    with pytest.raises(ValueError, match="Invalid parameter"):
        NWSignatureRegressor().set_params(bandwith=0.1)


# --------------------------------------------------------------------------- inputs

def test_initial_condition(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="requires X0"):
        NWSignatureRegressor(initial_condition=True).fit(X, y)
    model = NWSignatureRegressor(bandwidth=0.1).fit(X, y)
    with pytest.warns(UserWarning, match="X0 is ignored"):
        model.predict(X, X0=X[:, 0, :])


def test_initial_condition_is_shift_invariant(regression_data):
    """Shifting all initial conditions by the same constant must not change predictions."""
    X, y = regression_data
    x0 = np.random.default_rng(1).normal(size=(80, 1))
    kwargs = dict(signature_level=3, robust=True, robust_C=2, robust_a=1, bandwidth=0.05, initial_condition=True)
    base = NWSignatureRegressor(**kwargs).fit(X[:60], y[:60], X0=x0[:60]).predict(X[60:], X0=x0[60:])
    shifted = NWSignatureRegressor(**kwargs).fit(X[:60], y[:60], X0=x0[:60] + 1).predict(X[60:], X0=x0[60:] + 1)
    np.testing.assert_allclose(shifted, base, atol=1e-4)


def test_old_argument_order_raises(regression_data):
    X, y = regression_data
    with pytest.raises(TypeError, match="old argument order"):
        NWSignatureRegressor().fit(X, X[:, 0, :], y)


@pytest.mark.parametrize("robust", [False, True])
def test_precomputed_signatures_equal_paths(regression_data, robust):
    X, y = regression_data
    kwargs = dict(signature_level=3, bandwidth=0.1, robust=robust, robust_C=2, robust_a=1)
    from_paths = NWSignatureRegressor(**kwargs).fit(X[:60], y[:60]).predict(X[60:])
    S = ii.sig(X, 3)
    from_sigs = NWSignatureRegressor(input_type="signatures", **kwargs).fit(S[:60], y[:60]).predict(S[60:])
    np.testing.assert_allclose(from_sigs, from_paths, rtol=1e-6)


def test_log_signature(regression_data):
    X, y = regression_data
    model = NWSignatureRegressor(signature_level=3, log_signature=True, bandwidth=0.1).fit(X, y)
    assert model.X_train_.shape[1] == ii.logsiglength(2, 3)
    assert np.all(np.isfinite(model.predict(X)))


def test_signature_mode_is_deprecated(regression_data):
    X, y = regression_data
    base = NWSignatureRegressor(signature_level=3, bandwidth=0.1).fit(X, y).predict(X)
    for mode in (0, 1, 2):
        with pytest.warns(FutureWarning):
            model = NWSignatureRegressor(signature_level=3, bandwidth=0.1, signature_mode=mode).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), base)


@pytest.mark.parametrize("kwargs", [
    dict(robust=True, robust_C=0.5),
    dict(robust=True, robust_C_grid=[0.5, 2]),
    dict(robust=True, robust_a=0),
    dict(robust=True, log_signature=True),
    dict(bandwidth=0),
    dict(bandwidth_grid=[]),
    dict(signature_level_grid=[0, 1]),
    dict(metric_choice="supremum"),
    dict(metric_choice="pvar", p_param=0.5),
    dict(input_type="signatures", metric_choice="l2"),
    dict(cv_folds=1),
])
def test_invalid_parameters_raise_at_fit(regression_data, kwargs):
    X, y = regression_data
    model = NWSignatureRegressor(**kwargs)  # construction never validates (scikit-learn convention)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            model.fit(X, y)
