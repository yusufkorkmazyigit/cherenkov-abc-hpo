"""Sensitivity check: random photon-level split vs. contiguous-block split.

The unbalanced file appears to be stored in simulation order (label runs are far fewer than
in a shuffled file), so photons from the same event are likely neighbours. Without event IDs,
splitting into contiguous blocks keeps most events on one side of the split. If accuracy is
similar under both splits, photon-level leakage between train and test is not inflating results.
"""
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

d = pd.read_csv("data/Unbalanced1MDataset.csv")
X = d[["PMTno", "Time", "Energy"]].to_numpy(float); y = d["isCherenkov"].to_numpy()
models = {"XGBoost": lambda: xgb.XGBClassifier(tree_method="hist", n_jobs=2, random_state=0, verbosity=0),
          "LightGBM": lambda: lgb.LGBMClassifier(n_jobs=2, random_state=0, verbose=-1)}
rows = []
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
for name, make in models.items():
    rows.append({"Split": "Random (stratified)", "Model": name, "Test accuracy": accuracy_score(yte, make().fit(Xtr, ytr).predict(Xte))})
# 5 contiguous folds: each 20% block serves once as the test set
blocks = np.array_split(np.arange(len(y)), 5)
for name, make in models.items():
    accs = []
    for b in blocks:
        mask = np.zeros(len(y), bool); mask[b] = True
        accs.append(accuracy_score(y[mask], make().fit(X[~mask], y[~mask]).predict(X[mask])))
    rows.append({"Split": "Contiguous blocks (5-fold)", "Model": name,
                 "Test accuracy": float(np.mean(accs)), "Std over folds": float(np.std(accs))})
out = pd.DataFrame(rows)
out.to_csv("results/leakage_check.csv", index=False)
print(out.to_markdown(index=False))
