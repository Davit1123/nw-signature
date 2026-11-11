"""
Benchmark: does learning the distance between signatures improve classification?

Uses the datasets, preprocessing and signature levels of
01_classification_examples.ipynb (time augmentation, MinMaxScaler fitted on the
training data, truncation level K from suggest_truncation_level) and compares

    nw_kfold  NWSignatureClassifier as in notebook 01: unweighted signature
              metric, bandwidth by 5-fold CV accuracy over linspace(1e-5, 2, 50)
    nw_loo    NWLearnedDistanceClassifier with max_iter=0: unweighted metric,
              bandwidth by leave-one-out cross-entropy (the initial model of
              the learned variants)
    levels    NWLearnedDistanceClassifier(distance='levels'): one learned
              weight per signature level
    proj_r    NWLearnedDistanceClassifier(distance='projection',
              projection_dim=r) for r = 2, 5, 10, 20

on the fixed train/test splits. Run from the repository root:

    python examples/learned_distance_benchmark.py [--datasets Libras ...] [--models levels ...]

Results are printed and merged into examples/results/learned_distance.csv
(rows of the same dataset and model are replaced).
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nwsig import NWLearnedDistanceClassifier, NWSignatureClassifier, augment_with_time, suggest_truncation_level

DATASETS = ["ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "Cricket", "Epilepsy",
            "EthanolConcentration", "ERing", "FaceDetection", "FingerMovements", "HandMovementDirection",
            "Handwriting", "Heartbeat", "Libras", "LSST", "NATOPS", "PenDigits", "RacketSports",
            "SelfRegulationSCP1", "SelfRegulationSCP2", "StandWalkJump", "UWaveGestureLibrary"]

MODELS = {
    "nw_kfold": lambda K: NWSignatureClassifier(input_type="paths", bandwidth_grid=np.linspace(0.00001, 2.0, 50),
                                                signature_level=K),
    "nw_loo": lambda K: NWLearnedDistanceClassifier(signature_level=K, max_iter=0),
    "levels": lambda K: NWLearnedDistanceClassifier(signature_level=K, distance="levels"),
    **{f"proj_r{r}": (lambda K, r=r: NWLearnedDistanceClassifier(signature_level=K, distance="projection",
                                                                 projection_dim=r))
       for r in (2, 5, 10, 20)},
}


def load(name):
    """Train/test data of one dataset, preprocessed as in notebook 01."""
    X_tr, y_tr = load_classification(name, split="train")
    X_te, y_te = load_classification(name, split="test")
    X_tr, y_tr = shuffle(X_tr, y_tr, random_state=1)
    X_te, y_te = shuffle(X_te, y_te, random_state=1)

    label_encoder = LabelEncoder()
    y_tr = label_encoder.fit_transform(y_tr)
    y_te = label_encoder.transform(y_te)

    X_tr_t, X_te_t = augment_with_time(X_tr), augment_with_time(X_te)
    scaler = MinMaxScaler()
    X_tr_t = scaler.fit_transform(X_tr_t.reshape(-1, X_tr_t.shape[-1])).reshape(X_tr_t.shape)
    X_te_t = scaler.transform(X_te_t.reshape(-1, X_te_t.shape[-1])).reshape(X_te_t.shape)

    K = np.min([suggest_truncation_level(n_obs=X_tr.shape[0], length=X_tr.shape[2], dim=X_tr.shape[1],
                                         max_sig_dim_total=5e6) - 1, 5])
    return X_tr_t, y_tr, X_te_t, y_te, int(np.max([K, 1]))


def evaluate(name, models):
    X_tr, y_tr, X_te, y_te, K = load(name)
    rows = []
    for label in models:
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = MODELS[label](K).fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        row = dict(dataset=name, model=label, K=K, n_train=len(y_tr),
                   accuracy=accuracy_score(y_te, model.predict(X_te)),
                   log_loss=log_loss(y_te, proba, labels=model.classes_),
                   fit_seconds=time.time() - start)
        if isinstance(model, NWLearnedDistanceClassifier):
            row.update(loo_init=model.loo_loss_init_, loo_final=model.loo_loss_, n_iter=model.n_iter_)
            if model.distance == "levels":
                row.update(weights=" ".join(f"{w:.3g}" for w in model.level_weights_))
        rows.append(row)
        print(f"{name:26s} {label:9s} acc={row['accuracy']:.4f} log_loss={row['log_loss']:.4f} "
              f"({row['fit_seconds']:.1f}s)" + (f"  LOO {row['loo_init']:.4f} -> {row['loo_final']:.4f}"
                                                 if "loo_init" in row else "")
              + (f"  weights {row['weights']}" if "weights" in row else ""), flush=True)
    return rows


def summary(results, models):
    acc = results.pivot(index="dataset", columns="model", values="accuracy")[models]
    print("\nTest accuracy\n" + acc.round(4).to_string())
    print("\nMean accuracy: " + ", ".join(f"{m}={acc[m].mean():.4f}" for m in models))
    for a, b in [(m, n) for i, m in enumerate(models) for n in models[i + 1:]]:
        diff = acc[b] - acc[a]
        line = (f"{b} vs {a}: {int((diff > 0).sum())} wins, {int((diff == 0).sum())} ties, "
                f"{int((diff < 0).sum())} losses, mean difference {diff.mean():+.4f}")
        if (diff != 0).sum() >= 1:
            line += f", Wilcoxon p={wilcoxon(acc[b], acc[a]).pvalue:.3f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "learned_distance.csv"))
    args = parser.parse_args()

    results = pd.DataFrame([row for name in args.datasets for row in evaluate(name, args.models)])
    if os.path.exists(args.out):
        previous = pd.read_csv(args.out)
        keep = ~previous.set_index(["dataset", "model"]).index.isin(results.set_index(["dataset", "model"]).index)
        results = pd.concat([previous[keep], results], ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results.to_csv(args.out, index=False)
    models = [m for m in MODELS if m in set(results.model)]
    summary(results[results.dataset.isin(args.datasets)], models)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
