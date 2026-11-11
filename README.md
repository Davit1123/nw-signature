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
[*Local regression on path spaces with signature metrics*](https://arxiv.org/abs/2510.16728).

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

```bibtex
@article{bayer2025local,
  title   = {Local regression on path spaces with signature metrics},
  author  = {Bayer, Christian and Gogolashvili, Davit and Pelizzari, Luca},
  journal = {arXiv preprint arXiv:2510.16728},
  year    = {2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.
