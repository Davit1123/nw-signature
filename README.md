# Nadaraya-Watson Signature Estimator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Non-parametric regression and classification on path and time-series data with
the Nadaraya-Watson estimator. The prediction at a path `x` is a kernel-weighted
average of the training targets,

```
m(x) = sum_i w_i(x) y_i,    w_i(x) = exp(-d(x, x_i)^2 / h) / sum_j exp(-d(x, x_j)^2 / h),
```

where the distance `d` is a signature metric (the distance between truncated
path signatures, optionally robustly normalised) or a classical path metric.
The method is described in
[*Local Regression on Path Spaces with Signature Metrics*](https://proceedings.mlr.press/v300/bayer26a.html)
(AISTATS 2026).

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/Davit1123/nw-signature.git
```

To run the example notebooks or the tests, clone the repository and install it
with the optional dependencies:

```bash
git clone https://github.com/Davit1123/nw-signature.git
cd nw-signature
pip install -e ".[examples,test]"
```

The extra `kernel` installs `sigkernel`, which is only needed for
`nwsig.metrics.full_sig_metric`.

## Quick Start

### Classification

```python
from nwsig import NWSignatureClassifier
import numpy as np

# Paths: shape (n_samples, n_timesteps, n_features)
X_train = np.random.randn(100, 50, 3).cumsum(axis=1)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.randn(20, 50, 3).cumsum(axis=1)

classifier = NWSignatureClassifier(signature_level=2, cv_folds=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
proba = classifier.predict_proba(X_test)   # columns follow classifier.classes_
```

### Regression

```python
from nwsig import NWSignatureRegressor

y_train = X_train[:, -1, 0] + np.random.randn(100) * 0.1

regressor = NWSignatureRegressor(signature_level=2, cv_folds=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(regressor.best_bandwidth_)            # bandwidth chosen by cross-validation
```

The estimators follow the scikit-learn API, so they work with tools such as
`cross_val_score`, `GridSearchCV` and `clone`.

### Robust signatures and path metrics

```python
# Robust (tensor-normalised) signatures, with C and a chosen by cross-validation
robust = NWSignatureRegressor(signature_level=3, robust=True).fit(X_train, y_train)

# Classical path metrics: 'supremum_metric', 'l1', 'l2', 'lp', 'var1', 'pvar'
pvar = NWSignatureRegressor(metric_choice="pvar", p_param=2).fit(X_train, y_train)
```

### Including the initial condition

Signatures are invariant to translations of a path, so by default the starting
value of each path is ignored. To include it, set `initial_condition=True` and
pass the initial conditions `X0`. The distance then becomes
`|x0 - y0| + rho(x, y)`, where `rho` is the chosen path metric:

```python
regressor = NWSignatureRegressor(signature_level=2, initial_condition=True)
regressor.fit(X_train, y_train, X0=X_train[:, 0, :])
y_pred = regressor.predict(X_test, X0=X_test[:, 0, :])
```

### Learned signature distances (experimental)

`NWLearnedDistanceRegressor` and `NWLearnedDistanceClassifier` use the same
(robust) signature representation `z` but learn the distance between
signatures by minimising the leave-one-out loss of the estimator (squared error
for regression, cross-entropy for classification), with Gaussian weights
`exp(-d^2 / (2 h^2))`:

- `distance="levels"`: one weight per signature level,
  `d^2 = sum_k exp(eta_k) |z^(k)(x) - z^(k)(x')|^2`, initialised at 1.
- `distance="projection"`: a linear projection `B` with `projection_dim` rows,
  `d^2 = |B (z(x) - z(x'))|^2`, initialised with the top principal directions
  of the training features.

```python
from nwsig import NWLearnedDistanceClassifier

model = NWLearnedDistanceClassifier(signature_level=3, distance="levels").fit(X_train, y_train)
print(model.level_weights_, model.loo_loss_init_, model.loo_loss_)

model = NWLearnedDistanceClassifier(signature_level=3, distance="projection", projection_dim=5).fit(X_train, y_train)
print(model.projection_.shape)
```

The projection has `projection_dim * n_features` parameters and can drive its
leave-one-out loss to almost zero for any labels, including random ones (with
the random labels of the example above it goes from about 0.70 to 0.017 or less
in five random draws), so it overfits. With at least as many signature features as training samples it can
place the training samples at arbitrary points. The level weights, with one
parameter per level, barely move on random labels.

Note that `bandwidth` is the length scale `h` of `exp(-d^2 / (2 h^2))` here,
not the `h` of `exp(-d^2 / h)` used by `NWSignatureRegressor`. With
`max_iter=0` the estimator is the unweighted one with `h` chosen by
leave-one-out. `examples/learned_distance_benchmark.py` compares the variants
on the classification datasets of notebook 01.

### Windowed and learned multi-scale distances (experimental)

The signature of the whole path is a global summary of it. These additions also
compare paths locally in time, on windows that are the same for every path:

- `windowed_signature_features(X, level, n_windows, overlap, dilation, window_weight)`
  returns features whose Euclidean distance is
  `d^2 = |S(x) - S(y)|^2 + mu * (1/J) * sum_j |S_j(x) - S_j(y)|^2`, with `S_j`
  the signature of window `j` dilated by `lambda` (level `k` times
  `lambda^k`); pass them to the estimators with `input_type='signatures'`.
- `WindowTree` chooses the windows by greedy binary splitting, with the
  leave-one-out Brier score of the estimator as the splitting criterion.
- `NWLearnedWindowClassifier` learns the distance. Its windows are the whole
  path, its halves, quarters and eighths and the windows shifted by half their
  width; each is represented by its increments and signed (Levy) areas, which
  follow from prefix sums of the path in O(T). Levels 3..`level` of the
  signature of the whole path are added through the equivariance
  `S^k(M x) = M^(⊗k) S^k(x)`, so the signature of `x` is computed once. One
  weight per (scale, increments or areas) and per level, and a linear channel
  map `M`, are learned by Adam on the leave-one-out Brier score, with early
  stopping on 20% of the training samples; the bandwidth is then selected by
  5-fold CV accuracy.

```python
from nwsig import NWLearnedWindowClassifier, augment_with_time

# channel 0 of augment_with_time(...) is time, which is passed through unprojected
model = NWLearnedWindowClassifier(level=3, time_channel=0).fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.block_weights_, model.projection_.shape)
```

`examples/sliding_window_benchmark.py` compares them on the 21 datasets of
notebook 01 (results in `examples/results/sliding_window.csv`):

| Model | Mean test accuracy | Fit time, all datasets |
|---|---|---|
| signature metric, as in notebook 01 | 0.622 | 26 s |
| windows, CV grid over their number, overlap, dilation and weight | 0.675 | 1513 s |
| the same with the windows at level 1 (increments only) | 0.667 | 372 s |
| `WindowTree` | 0.638 | 986 s |
| `NWLearnedWindowClassifier`, weights learned | 0.664 | 117 s |
| `NWLearnedWindowClassifier`, weights and channel map learned | 0.666 | 175 s |

Against the signature metric, the CV grid of windows wins on 15 datasets and
loses on 5 (Wilcoxon p = 0.002), the learned classifier wins on 15 and loses
on 4 (p = 0.003); the two do not differ significantly (p = 0.55). The time of
the grid was measured while the machine was swapping. The variants were
compared on the same test splits several times, and the bandwidth selection of
the learned classifier was chosen after seeing them, so the comparison should
be confirmed on other datasets.

## Hyperparameter selection

Every hyperparameter that you fix is used as given; every one left as `None` is
selected by K-fold cross-validation (mean squared error for regression,
accuracy for classification) from its grid:

| Hyperparameter | Fixed value | Grid (default) |
|---|---|---|
| bandwidth | `bandwidth` | `bandwidth_grid` (0.001, 0.005, 0.01, 0.05, 0.1, 0.5) |
| signature level | `signature_level` | `signature_level_grid` (none: `signature_level` is used) |
| robust C | `robust_C` | `robust_C_grid` (1, 2, 5, 10) |
| robust a | `robust_a` | `robust_a_grid` (0.5, 1, 1.5, 2) |

The folds are shuffled with `random_state` and stratified by class for
classification. Pass `verbose=True` to print the cross-validation results.

## Key Parameters

- `signature_level` (int): signature truncation level (default: 2)
- `metric_choice` (str): `'signature_metric'` (default), `'supremum_metric'`, `'l1'`, `'l2'`, `'lp'`, `'var1'` or `'pvar'`
- `p_param` (float): exponent of the `'pvar'` and `'lp'` metrics (default: 2)
- `robust` (bool): use robust signatures (default: False); `robust_C` must be >= 1 and `robust_a` > 0
- `log_signature` (bool): use log-signatures (default: False)
- `input_type` (str): `'paths'` (default) or `'signatures'` for pre-computed signatures
- `initial_condition` (bool): include the initial condition `X0` in the distance (default: False)
- `cv_folds` (int), `random_state` (int): cross-validation folds (default: 5) and seed (default: 42)
- `use_gpu` (bool or None): use CUDA if available (default: None, i.e. automatic)
- `verbose` (bool): print progress (default: False)

See the docstring of `NadarayaWatsonSignature` for the full list.

## Input Format

```python
X: shape (n_samples, n_timesteps, n_features)  # paths (or (n_samples, signature_dim) for input_type='signatures')
y: shape (n_samples,)                           # targets or class labels
X0: shape (n_samples, n_features)               # initial conditions, only with initial_condition=True
```

## Examples

The `examples/` directory contains four Jupyter notebooks:

1. **01_classification_examples.ipynb** - Classification on 20+ benchmark datasets
2. **02_regression_examples.ipynb** - Regression with baseline comparisons
3. **03_learning_sde_solution.ipynb** - Learning SDE solution maps
4. **04_path_dependent_options.ipynb** - Path-dependent option pricing

Run the examples:
```bash
cd examples
jupyter notebook
```

## Running the tests

From the repository root, after installing with the `test` extra:

```bash
pytest
```

The tests compare the implementation with independent references: the robust
signature against its definition, p-variation against a brute-force
computation, cross-validation against a fold-by-fold evaluation, and the
fractional Brownian motion simulators against the theoretical covariance.

## Dependencies

- numpy, scipy, scikit-learn
- torch (for GPU acceleration)
- iisignature (signature computation)
- numba (JIT compilation)
- aeon (for example datasets)
- sigkernel (optional, only for `nwsig.metrics.full_sig_metric`)

See `requirements.txt` for the full list.

## Citation

If you use this code, please cite:

```bibtex
@InProceedings{pmlr-v300-bayer26a,
  title     = {Local Regression on Path Spaces with Signature Metrics},
  author    = {Bayer, Christian and Gogolashvili, Davit and Pelizzari, Luca},
  booktitle = {Proceedings of The 29th International Conference on Artificial Intelligence and Statistics},
  pages     = {2755--2763},
  year      = {2026},
  editor    = {Khan, Emtiyaz and Li, Yingzhen and Solin, Arno and Ramdas, Aaditya},
  volume    = {300},
  series    = {Proceedings of Machine Learning Research},
  month     = {02--05 May},
  publisher = {PMLR},
  pdf       = {https://raw.githubusercontent.com/mlresearch/v300/main/assets/bayer26a/bayer26a.pdf},
  url       = {https://proceedings.mlr.press/v300/bayer26a.html}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.
