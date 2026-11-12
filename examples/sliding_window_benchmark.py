"""
Benchmark: do sliding-window signatures improve Nadaraya-Watson classification?

Uses the datasets, preprocessing and signature levels of
01_classification_examples.ipynb (time augmentation, MinMaxScaler fitted on
the training data, truncation level K from suggest_truncation_level) and
compares, on the fixed train/test splits,

    nb_grid     signature metric |S(x) - S(y)| with the bandwidth grid of
                notebook 01, linspace(1e-5, 2, 50) (reproduces nw_kfold of
                learned_distance_benchmark.py)
    global      signature metric, bandwidth grid relative to the distances
    global_dil  signature of the dilated path lambda * x, lambda in {1, 2, 4}:
                rescaling the levels without windows, a control
    window      global + mu * windows,
                    d^2 = |S(x) - S(y)|^2 / m_g + mu * sum_j |S_j(x) - S_j(y)|^2 / m_w,
                with S_j the signature of window j dilated by lambda = J^gamma,
                windows of width 1/J of the path (J in {2, 4, 8, 16}), without
                or with half overlap (J or 2J - 1 windows), gamma in {0, 1/2, 1}
                and mu in {1/4, 1, 4, inf} (inf: windows only); m_g and m_w
                are the median squared distances between training paths, so mu
                is the relative weight of the windows
    auto        global or window, whichever has the higher CV accuracy
    window_l1   window with the windows at signature level 1 (their increments)
                and the global signature at level K: is the gain the signatures
                of the windows, or the localisation in time?
    tree        windows chosen by greedy binary splitting (nwsig.WindowTree with
                its defaults: splits by the leave-one-out Brier score); the
                bandwidth is then selected by CV accuracy like for the others
    tree_ce     the same with splits by the leave-one-out cross-entropy
    learned_w   nwsig.NWLearnedWindowClassifier with the channel map fixed:
                weights of the multi-scale windows (increments, areas) and of
                the levels 3..K of the whole path learned by gradient descent
                on the leave-one-out Brier score, early stopping on 20% of the
                training samples; the bandwidth then by 5-fold CV accuracy
    learned     the same with the linear channel map learned too
    learned_ens average of the class probabilities of learned_w and learned
    learned_w_brier, learned_brier
                the same with the bandwidth minimising the Brier score on the
                20% validation samples instead (no CV at all)
    tree_random WindowTree on randomly permuted training labels: the reduction
                of its leave-one-out loss is what the splitting gains on noise
                (no test accuracy; only for at most RANDOM_LABEL_MAX_TRAIN
                training paths, where overfitting the cuts is the risk)

Every hyperparameter is selected by the 5-fold CV accuracy of
NadarayaWatsonSignature (folds stratified and shuffled with random_state=42,
weights softmax(-d^2 / h), ties to the first candidate in grid order). Except
for nb_grid, the bandwidth is h = c * m with m the median squared distance
between training paths and c in logspace(-4, 1, 41), from (in effect) the
nearest neighbour to uniform weights, so that every representation gets a grid
on its own scale. The CV runs on precomputed distance matrices, which gives
the same predictions as the estimators (up to float32 vs float64).

Run from the repository root:

    python examples/sliding_window_benchmark.py [--datasets Libras ...] [--models tree ...] [--data-dir ~/aeon_data]

Results are printed and merged into examples/results/sliding_window.csv after
every dataset (rows of the same dataset and model are replaced).
"""
import argparse
import os
import sys
import time
import warnings

import iisignature as ii
import numpy as np
import pandas as pd
import torch
from aeon.datasets import load_classification
from scipy.stats import wilcoxon
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nwsig import (NWLearnedWindowClassifier, NWSignatureClassifier, WindowTree, augment_with_time,
                   sliding_window_signatures, suggest_truncation_level)

DATASETS = ["ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "Cricket", "Epilepsy",
            "EthanolConcentration", "ERing", "FaceDetection", "FingerMovements", "HandMovementDirection",
            "Handwriting", "Heartbeat", "Libras", "LSST", "NATOPS", "PenDigits", "RacketSports",
            "SelfRegulationSCP1", "SelfRegulationSCP2", "StandWalkJump", "UWaveGestureLibrary"]
MODELS = ["nb_grid", "global", "global_dil", "window", "auto", "window_l1", "tree", "tree_ce", "tree_random", "learned_w", "learned", "learned_ens", "learned_w_brier",
          "learned_brier"]

NOTEBOOK_BANDWIDTHS = np.linspace(0.00001, 2.0, 50)
BANDWIDTH_FACTORS = np.logspace(-4, 1, 41)
GLOBAL_DILATIONS = (1.0, 2.0, 4.0)
SEGMENTS = (2, 4, 8, 16)
OVERLAPS = (0.0, 0.5)
GAMMAS = (0.0, 0.5, 1.0)
MUS = (0.25, 1.0, 4.0, np.inf)
DTYPE = torch.float64
RANDOM_LABEL_MAX_TRAIN = 1000


def load(name, extract_path=None):
    """Train/test data of one dataset, preprocessed as in notebook 01 (and learned_distance_benchmark.py)."""
    X_tr, y_tr = load_classification(name, split="train", extract_path=extract_path)
    X_te, y_te = load_classification(name, split="test", extract_path=extract_path)
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


def sq_dists(A, B=None):
    """Squared Euclidean distances between the rows of A and B (default A), float64, built in place."""
    A = torch.as_tensor(A, dtype=DTYPE)
    B = A if B is None else torch.as_tensor(B, dtype=DTYPE)
    D2 = A @ B.T
    D2.mul_(-2.0).add_((A * A).sum(1)[:, None]).add_((B * B).sum(1)[None, :])
    return D2.clamp_min_(0.0)


def pair_median(D2, n_pairs=200_000, seed=0):
    """Median of D2[i, j] over (a random sample of) the pairs i != j."""
    n = D2.shape[0]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = (i + rng.integers(1, n, n_pairs)) % n             # j != i
    return float(torch.median(D2[torch.as_tensor(i), torch.as_tensor(j)]))


class CrossValidation:
    """5-fold CV accuracy of the Nadaraya-Watson classifier on precomputed squared distances."""

    def __init__(self, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)    # non-stratified fallback for tiny classes
            folds = NWSignatureClassifier()._cv_folds(y)    # the folds of the estimator
        self.folds = [(torch.as_tensor(tr), torch.as_tensor(va)) for tr, va in folds]
        self.y = torch.as_tensor(y)
        self.onehot = torch.nn.functional.one_hot(self.y).to(DTYPE)

    def accuracy(self, D2, bandwidths):
        """Mean validation accuracy for every bandwidth."""
        acc = np.zeros(len(bandwidths))
        for tr, va in self.folds:
            sub, Y, y_va = D2[va][:, tr], self.onehot[tr], self.y[va]
            for b, h in enumerate(bandwidths):
                pred = (torch.softmax(-sub / h, dim=1) @ Y).argmax(dim=1)
                acc[b] += (pred == y_va).double().mean().item()
        return acc / len(self.folds)


def predict(D2_te_tr, onehot_tr, h):
    return (torch.softmax(-D2_te_tr / h, dim=1) @ onehot_tr).argmax(dim=1).numpy()


def n_windows(J, overlap):
    """Number of windows of width 1/J of the path that start (1 - overlap) / J apart."""
    return int(round((J - 1) / (1 - overlap))) + 1


class Dataset:
    """Distance matrices of one dataset for the representations of the benchmark."""

    def __init__(self, X_tr, y_tr, X_te, y_te, K):
        self.X_tr, self.X_te, self.y_tr, self.y_te, self.K = X_tr, X_te, y_tr, y_te, K
        self.cv = CrossValidation(y_tr)
        self.onehot = self.cv.onehot
        self.T = X_tr.shape[1]
        S_tr, S_te = ii.sig(X_tr, K), ii.sig(X_te, K)
        # squared signature distances, unnormalised: the relative bandwidth grid
        # does not depend on their scale, the notebook grid of nb_grid does
        self.G = sq_dists(S_tr)
        self.G_te = sq_dists(S_te, S_tr)
        self.m_g = pair_median(self.G)

    def global_dilated(self, lam, test=False):
        if lam == 1.0:
            return self.G_te if test else self.G
        S_tr = ii.sig(lam * self.X_tr, self.K)
        return sq_dists(ii.sig(lam * self.X_te, self.K), S_tr) if test else sq_dists(S_tr)

    def windows(self, J, overlap, gamma, test=False, level=None):
        """
        Squared window distances over their training median (train-train, or
        test-train if test), with window signatures at level (default K).
        """
        args = dict(level=level or self.K, n_windows=n_windows(J, overlap), overlap=overlap,
                    dilation=float(J) ** gamma)
        W_tr = sliding_window_signatures(self.X_tr, **args).reshape(len(self.X_tr), -1)
        D2 = sq_dists(W_tr)
        m = pair_median(D2)
        if test:
            W_te = sliding_window_signatures(self.X_te, **args).reshape(len(self.X_te), -1)
            D2 = sq_dists(W_te, W_tr)
        return D2.div_(m)

    def combine(self, Wd, mu, test=False):
        """G / m_g + mu * Wd (Wd itself for mu = inf)."""
        if np.isinf(mu):
            return Wd
        return (Wd * mu).add_(self.G_te if test else self.G, alpha=1.0 / self.m_g)


def window_configs(K, T):
    """Grid of the window model, in tie-breaking order."""
    gammas = GAMMAS if K > 1 else (0.0,)                 # at level 1 the dilation is a rescaling of mu
    for J in SEGMENTS:
        if (T - 1) / J < 2:                              # windows of at least two time steps
            continue
        for overlap in OVERLAPS:
            for gamma in gammas:
                yield J, overlap, gamma


def select(cv, candidates, factors=BANDWIDTH_FACTORS, relative=True):
    """
    Best (CV accuracy, config, bandwidth) over candidates (config, D2_train),
    ties to the first; bandwidth h = c * median(D2) with c in factors if relative.
    """
    best = None
    for config, D2 in candidates:
        scale = pair_median(D2) if relative else 1.0
        acc = cv.accuracy(D2, factors * scale)
        b = int(np.argmax(acc))
        if best is None or acc[b] > best[0]:
            best = (acc[b], dict(config, c=factors[b]), factors[b] * scale)
    return best


def describe_tree(tree):
    """Windows, weights and leave-one-out losses of a fitted WindowTree, for the results."""
    return dict(gamma=tree.gamma_, n_windows=len(tree.windows_), loo_init=tree.loo_loss_init_,
                loo_final=tree.loo_loss_, loo_gain=1.0 - tree.loo_loss_ / tree.loo_loss_init_,
                windows=" ".join(f"{a}-{b}" for a, b in tree.windows_),
                weights=" ".join(f"{w:g}" for w in tree.weights_))


def evaluate(name, models, extract_path=None):
    start = time.time()
    data = Dataset(*load(name, extract_path))
    K, T, cv = data.K, data.T, data.cv
    base = dict(dataset=name, K=K, T=T, n_train=len(data.X_tr), n_test=len(data.X_te))
    rows = {}

    def record(model, cv_acc, config, h, D2_te, t0):
        acc = float(np.mean(predict(D2_te, data.onehot, h) == data.y_te))
        rows[model] = dict(base, model=model, cv_accuracy=cv_acc, accuracy=acc, seconds=time.time() - t0, **config)
        desc = " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in config.items())
        print(f"{name:26s} {model:11s} cv={cv_acc:.4f} test={acc:.4f}  {desc}", flush=True)

    if "nb_grid" in models:
        t0 = time.time()
        cv_acc, config, h = select(cv, [({}, data.G)], NOTEBOOK_BANDWIDTHS, relative=False)
        config = {"h": config.pop("c")}
        record("nb_grid", cv_acc, config, h, data.G_te, t0)

    if "global" in models or "auto" in models:
        t0 = time.time()
        cv_acc, config, h = select(cv, [({}, data.G)])
        record("global", cv_acc, config, h, data.G_te, t0)

    if "global_dil" in models:
        t0 = time.time()
        dilations = GLOBAL_DILATIONS if K > 1 else (1.0,)
        cv_acc, config, h = select(cv, (({"lam": lam}, data.global_dilated(lam)) for lam in dilations))
        record("global_dil", cv_acc, config, h, data.global_dilated(config["lam"], test=True), t0)

    def window_model(model, level):
        t0 = time.time()

        def candidates():
            for J, overlap, gamma in window_configs(level, T):
                Wd = data.windows(J, overlap, gamma, level=level)
                for mu in MUS:
                    yield dict(J=J, overlap=overlap, gamma=gamma, mu=mu), data.combine(Wd, mu)

        cv_acc, config, h = select(cv, candidates())
        Wd_te = data.windows(config["J"], config["overlap"], config["gamma"], test=True, level=level)
        record(model, cv_acc, config, h, data.combine(Wd_te, config["mu"], test=True), t0)

    if "window" in models or "auto" in models:
        window_model("window", K)
    if "auto" in models:
        chosen = "window" if rows["window"]["cv_accuracy"] > rows["global"]["cv_accuracy"] else "global"
        rows["auto"] = dict(rows[chosen], model="auto", chosen=chosen)
        print(f"{name:26s} {'auto':11s} -> {chosen}", flush=True)
    if "window_l1" in models:
        window_model("window_l1", 1)

    for model, criterion in (("tree", "brier"), ("tree_ce", "cross_entropy")):
        if model in models:
            t0 = time.time()
            tree = WindowTree(level=K, criterion=criterion).fit(data.X_tr, data.y_tr)
            F_tr = tree.transform(data.X_tr)
            D2, D2_te = sq_dists(F_tr), sq_dists(tree.transform(data.X_te), F_tr)
            cv_acc, config, h = select(cv, [({}, D2)])
            record(model, cv_acc, dict(config, **describe_tree(tree)), h, D2_te, t0)

    if "tree_random" in models and len(data.y_tr) <= RANDOM_LABEL_MAX_TRAIN:
        t0 = time.time()
        y_random = np.random.default_rng(0).permutation(data.y_tr)
        tree = WindowTree(level=K).fit(data.X_tr, y_random)
        rows["tree_random"] = dict(base, model="tree_random", seconds=time.time() - t0, **describe_tree(tree))
        print(f"{name:26s} {'tree_random':11s} LOO {tree.loo_loss_init_:.4f} -> {tree.loo_loss_:.4f} "
              f"with {len(tree.windows_)} windows", flush=True)

    for model, learn_projection, bandwidth in (("learned_w", False, "cv_accuracy"), ("learned", True, "cv_accuracy"),
                                               ("learned_w_brier", False, "validation_brier"),
                                               ("learned_brier", True, "validation_brier")):
        if model in models:
            t0 = time.time()
            clf = NWLearnedWindowClassifier(level=max(K, 2), time_channel=0, learn_projection=learn_projection,
                                            bandwidth_selection=bandwidth).fit(data.X_tr, data.y_tr)
            fit_seconds = time.time() - t0
            t0 = time.time()
            acc = float(np.mean(clf.predict(data.X_te) == data.y_te))
            best = min(clf.history_, key=lambda h: h["validation"])
            rows[model] = dict(base, model=model, accuracy=acc, seconds=fit_seconds, cv_accuracy=clf.cv_accuracy_,
                               predict_seconds=time.time() - t0, n_iter=clf.n_iter_, level=clf.level_,
                               channels=clf.projection_.shape[0], val_accuracy=best["validation_accuracy"],
                               weights=" ".join(f"{w:.3g}" for w in clf.block_weights_.values()))
            print(f"{name:26s} {model:11s} test={acc:.4f}  fit {fit_seconds:.1f}s ({clf.n_iter_} steps), "
                  f"predict {rows[model]['predict_seconds']:.2f}s, level {clf.level_}, "
                  f"{clf.projection_.shape[0]} channels", flush=True)

    if "learned_ens" in models:
        t0 = time.time()
        clfs = [NWLearnedWindowClassifier(level=max(K, 2), time_channel=0, learn_projection=lp).fit(data.X_tr, data.y_tr)
                for lp in (False, True)]
        fit_seconds = time.time() - t0
        t0 = time.time()
        proba = np.mean([clf.predict_proba(data.X_te) for clf in clfs], axis=0)
        acc = float(np.mean(clfs[0].classes_[proba.argmax(axis=1)] == data.y_te))
        rows["learned_ens"] = dict(base, model="learned_ens", accuracy=acc, seconds=fit_seconds,
                                   predict_seconds=time.time() - t0)
        print(f"{name:26s} {'learned_ens':11s} test={acc:.4f}  fit {fit_seconds:.1f}s", flush=True)

    print(f"{name:26s} ({time.time() - start:.1f}s in total)", flush=True)
    return list(rows.values())


def summary(results, reference=None):
    acc = results.pivot(index="dataset", columns="model", values="accuracy").dropna(axis=1, how="all")
    models = [m for m in MODELS if m in acc]
    acc = acc[models]
    if reference is not None:
        acc.insert(0, "ref_nw_kfold", reference.reindex(acc.index))
    print("\nTest accuracy\n" + acc.round(4).to_string())
    print("\nMean accuracy: " + ", ".join(f"{m}={acc[m].mean():.4f}" for m in acc))
    pairs = ([("global", m) for m in models if m not in ("nb_grid", "global")]
             + [("window", "window_l1"), ("window", "tree"), ("tree_ce", "tree"), ("window", "learned_w"),
                ("window", "learned"), ("learned_w", "learned"), ("learned", "learned_ens"),
                ("window", "learned_ens"), ("learned_brier", "learned"),
                ("learned_w_brier", "learned_w"), ("nb_grid", "global")])
    if reference is not None:
        pairs.insert(0, ("ref_nw_kfold", "nb_grid"))
    for a, b in pairs:
        if a not in acc or b not in acc:
            continue
        diff = (acc[b] - acc[a]).dropna()
        line = (f"{b} vs {a}: {int((diff > 0).sum())} wins, {int((diff == 0).sum())} ties, "
                f"{int((diff < 0).sum())} losses, mean difference {diff.mean():+.4f}")
        if (diff != 0).sum() >= 1:
            line += f", Wilcoxon p={wilcoxon(diff).pvalue:.3f}"
        print(line)
    seconds = results.pivot(index="dataset", columns="model", values="seconds")
    print("\nFit time in seconds, total over the datasets (median per dataset): " + ", ".join(
        f"{m}={seconds[m].sum():.0f} ({seconds[m].median():.1f})" for m in models if m in seconds
        and m not in ("auto",)))
    cv = results.pivot(index="dataset", columns="model", values="cv_accuracy")
    for m in ("window", "tree", "tree_ce"):
        if {"global", m} <= set(cv) and m in acc:
            gain_cv, gain_test = cv[m] - cv["global"], acc[m] - acc["global"]
            print(f"{m} - global: mean CV gain {gain_cv.mean():+.4f}, mean test gain {gain_test.mean():+.4f}")
    if {"tree", "tree_random"} <= set(results.model):
        gain = results.pivot(index="dataset", columns="model", values="loo_gain")[["tree", "tree_random"]]
        size = results.pivot(index="dataset", columns="model", values="n_windows")[["tree", "tree_random"]]
        table = pd.concat([gain.add_prefix("loo_gain_"), size.add_prefix("n_windows_")], axis=1)
        print("\nRelative reduction of the leave-one-out loss by the tree, real vs random labels\n"
              + table.round(3).to_string())
        print(f"Mean: real {gain['tree'].mean():.3f}, random {gain['tree_random'].mean():.3f}")


def save(rows, path):
    """Merge rows into the CSV at path, replacing rows of the same dataset and model."""
    results = pd.DataFrame(rows)
    if os.path.exists(path):
        previous = pd.read_csv(path)
        keep = ~previous.set_index(["dataset", "model"]).index.isin(results.set_index(["dataset", "model"]).index)
        results = pd.concat([previous[keep], results], ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path, index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--data-dir", default=None,
                        help="directory with the datasets (aeon's extract_path); default: aeon's own")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "sliding_window.csv"))
    parser.add_argument("--threads", type=int, default=None, help="torch threads")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    data_dir = os.path.expanduser(args.data_dir) if args.data_dir else None

    for name in args.datasets:
        results = save(evaluate(name, args.models, data_dir), args.out)   # saved after every dataset

    reference = None
    ref_path = os.path.join(os.path.dirname(__file__), "results", "learned_distance.csv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        reference = ref[ref.model == "nw_kfold"].set_index("dataset")["accuracy"]
    summary(results[results.dataset.isin(args.datasets)], reference)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
