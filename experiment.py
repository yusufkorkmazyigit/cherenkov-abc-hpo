"""
Cherenkov / scintillation photon separation: hyperparameter optimization benchmark.

Compares default settings, Grid Search, Random Search, Optuna TPE and ABC (abcopt)
for XGBoost and LightGBM under an identical budget of candidate evaluations.

Protocol
- Features: PMTno, Time, Energy. Target: isCherenkov.
- Fixed stratified 80/20 train/test split (seed 0). The test set is used once per run.
- Search: each candidate is scored by 3-fold stratified CV accuracy on a fixed
  50,000-photon subsample of the training set (same subsample and folds for every method).
- The best configuration of each run is refit on the full training set and evaluated on the test set.
- Grid Search is deterministic, so it runs once; stochastic methods run with 5 seeds.

Results are appended to results/*.jsonl so the run can be resumed.
"""
import json
import os
import sys
import time
import warnings
from itertools import product

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)

from abcopt import ABCSearchCV

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = os.environ.get("DATA_DIR", "data")
OUT_DIR = "results"
FEATURES = ["PMTno", "Time", "Energy"]
TARGET = "isCherenkov"
BUDGET = 48
SEEDS = [0, 1, 2, 3, 4]
SEARCH_N = 50_000
N_JOBS = int(os.environ.get("N_JOBS", 2))
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

DATASETS = {"balanced": "Balanced700kDataset.csv", "unbalanced": "Unbalanced1MDataset.csv"}

# (name, type, low, high, log)
MODELS = {
    "XGBoost": dict(
        make=lambda seed: xgb.XGBClassifier(tree_method="hist", n_jobs=N_JOBS, random_state=seed, verbosity=0),
        space=[("learning_rate", "float", 0.01, 0.3, True),
               ("n_estimators", "int", 100, 600, False),
               ("max_depth", "int", 3, 12, False),
               ("min_child_weight", "float", 0.5, 20.0, True),
               ("subsample", "float", 0.5, 1.0, False),
               ("reg_lambda", "float", 1e-3, 10.0, True)],
        grid={"learning_rate": [0.03, 0.1, 0.3], "n_estimators": [200, 500],
              "max_depth": [4, 6, 8, 10], "min_child_weight": [1.0, 5.0]},
    ),
    "LightGBM": dict(
        make=lambda seed: lgb.LGBMClassifier(n_jobs=N_JOBS, random_state=seed, verbose=-1, subsample_freq=1),
        space=[("learning_rate", "float", 0.01, 0.3, True),
               ("n_estimators", "int", 100, 600, False),
               ("num_leaves", "int", 15, 255, False),
               ("min_child_samples", "int", 5, 100, False),
               ("subsample", "float", 0.5, 1.0, False),
               ("reg_lambda", "float", 1e-3, 10.0, True)],
        grid={"learning_rate": [0.03, 0.1, 0.3], "n_estimators": [200, 500],
              "num_leaves": [31, 63, 127, 255], "min_child_samples": [20, 60]},
    ),
}


def to_abc(space):
    return {n: {"type": t, "range": (lo, hi), "log": lg} for n, t, lo, hi, lg in space}


def to_sklearn(space):
    out = {}
    for n, t, lo, hi, lg in space:
        if t == "int":
            out[n] = randint(lo, hi + 1)
        elif lg:
            out[n] = loguniform(lo, hi)
        else:
            out[n] = uniform(lo, hi - lo)
    return out


def suggest(trial, space):
    p = {}
    for n, t, lo, hi, lg in space:
        p[n] = trial.suggest_int(n, lo, hi) if t == "int" else trial.suggest_float(n, lo, hi, log=lg)
    return p


def search(method, cfg, seed, Xs, ys):
    """Return (best_params, list_of_candidate_cv_scores)."""
    base = cfg["make"](seed)
    if method == "Default":
        return {}, [cross_val_score(base, Xs, ys, cv=CV, scoring="accuracy").mean()]
    if method == "Grid Search":
        grid = [dict(zip(cfg["grid"], v)) for v in product(*cfg["grid"].values())]
        assert len(grid) == BUDGET, len(grid)
        scores = [cross_val_score(clone(base).set_params(**g), Xs, ys, cv=CV, scoring="accuracy").mean() for g in grid]
        return grid[int(np.argmax(scores))], scores
    if method == "Random Search":
        s = RandomizedSearchCV(base, to_sklearn(cfg["space"]), n_iter=BUDGET, cv=CV, scoring="accuracy",
                               random_state=seed, refit=False).fit(Xs, ys)
        return s.best_params_, list(s.cv_results_["mean_test_score"])
    if method == "Optuna TPE":
        def obj(trial):
            m = clone(base).set_params(**suggest(trial, cfg["space"]))
            return cross_val_score(m, Xs, ys, cv=CV, scoring="accuracy").mean()
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        st.optimize(obj, n_trials=BUDGET)
        return st.best_params, [t.value for t in st.trials]
    if method == "ABC":
        s = ABCSearchCV(base, to_abc(cfg["space"]), cv=CV, scoring="accuracy", pop_size=8,
                        max_evals=BUDGET, random_state=seed, refit=False).fit(Xs, ys)
        return s.best_params_, [r["score"] for r in s.results_]
    raise ValueError(method)


def to_jsonable(d):
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def main(only=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    for dname, fname in DATASETS.items():
        if only and dname not in only:
            continue
        out_path = os.path.join(OUT_DIR, f"{dname}.jsonl")
        done = set()
        if os.path.exists(out_path):
            for line in open(out_path):
                r = json.loads(line)
                done.add((r["model"], r["method"], r["seed"]))
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        X, y = df[FEATURES].to_numpy(float), df[TARGET].to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=SEARCH_N, stratify=y_tr, random_state=0)
        print(f"[{dname}] train={len(y_tr):,} test={len(y_te):,} search subsample={len(ys):,}", flush=True)

        for mname, cfg in MODELS.items():
            for method in ["Default", "Grid Search", "Random Search", "Optuna TPE", "ABC"]:
                seeds = [0] if method in ("Default", "Grid Search") else SEEDS
                for seed in seeds:
                    if (mname, method, seed) in done:
                        continue
                    t0 = time.perf_counter()
                    params, scores = search(method, cfg, seed, Xs, ys)
                    t_search = time.perf_counter() - t0
                    model = clone(cfg["make"](seed)).set_params(**params).fit(X_tr, y_tr)
                    proba = model.predict_proba(X_te)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    rec = dict(dataset=dname, model=mname, method=method, seed=seed,
                               params=to_jsonable(params), cv_scores=[float(s) for s in scores],
                               best_cv=float(max(scores)), search_seconds=t_search,
                               test_accuracy=accuracy_score(y_te, pred), test_f1=f1_score(y_te, pred),
                               test_auc=roc_auc_score(y_te, proba), test_logloss=log_loss(y_te, proba))
                    with open(out_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"  {mname:<8} {method:<13} seed={seed} cv={rec['best_cv']:.5f} "
                          f"test_acc={rec['test_accuracy']:.5f} ({t_search:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
