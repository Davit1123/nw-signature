"""Shared fixtures for the test suite."""
import sys
from pathlib import Path

import numpy as np
import pytest

# The package is used from the repository root (it is not installed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_paths(n, n_timesteps=41, dim=2, scale=1.0, seed=0):
    """
    Time-augmented Brownian paths of shape (n, n_timesteps, dim): channel 0 is
    time on [0, 1], the other channels are Brownian motions started at 0.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_timesteps)
    increments = rng.normal(size=(n, n_timesteps - 1, dim - 1)) * scale / np.sqrt(n_timesteps - 1)
    W = np.concatenate([np.zeros((n, 1, dim - 1)), increments.cumsum(axis=1)], axis=1)
    return np.concatenate([np.broadcast_to(t[None, :, None], (n, n_timesteps, 1)), W], axis=2)


@pytest.fixture
def make_paths():
    return _make_paths


@pytest.fixture
def regression_data():
    X = _make_paths(80)
    y = X[:, -1, 1] + X[:, :, 1].max(axis=1)
    return X, y


@pytest.fixture
def classification_data():
    X = _make_paths(90)
    y = np.array(["down", "flat", "up"])[np.digitize(X[:, -1, 1], [-0.3, 0.3])]
    return X, y
