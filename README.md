# Nadaraya-Watson Signature Estimator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Implementation of Nadaraya-Watson estimators using path signatures for non-parametric regression and classification on time series and path data.

## Installation

```bash
git clone https://github.com/anonymous-review-submission/nw-signature.git
cd nw-signature
pip install -r requirements.txt
```

## Quick Start

### Classification

```python
from nwsig import NWSignatureClassifier
import numpy as np

# Your time series data: shape (n_samples, n_timesteps, n_features)
X_train = np.random.randn(100, 50, 3)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.randn(20, 50, 3)

# Fit and predict
classifier = NWSignatureClassifier(signature_level=2, cv_folds=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```

### Regression

```python
from nwsig import NWSignatureRegressor

regressor = NWSignatureRegressor(signature_level=2, cv_folds=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
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

## Key Parameters

- `signature_level` (int): Signature truncation level (default: 2)
- `bandwidth_grid` (list): Grid of bandwidth values for cross-validation
- `cv_folds` (int): Number of CV folds (default: 5)
- `robust` (bool): Use robust signature transformations (default: False)
- `metric_choice` (str): Distance metric - `'signature_metric'`, `'supremum_metric'`, `'l1'`, `'l2'`, `'pvar'`
- `initial_condition` (bool): Include the initial condition `X0` in the distance (default: False)

## Input Format

```python
X: shape (n_samples, n_timesteps, n_features)  # Path data
y: shape (n_samples,)                           # Target values
X0: shape (n_samples, n_features)               # Initial conditions, only with initial_condition=True
```

## Dependencies

- numpy, scipy, scikit-learn
- torch (for GPU acceleration)
- iisignature (signature computation)
- numba (JIT compilation)
- aeon (for example datasets)

See `requirements.txt` for full list.

## License

MIT License - see [LICENSE](LICENSE) file.
