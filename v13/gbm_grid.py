"""Grid search GBM hyperparameters (replaces made-up values)."""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)
FEATS = ["cos", "ent", "ana", "mass_err", "log_prior", "t_top1", "metfrag", "fpdot"]


def mrr_of(df):
    rr = []
    for _, g in df.groupby("truth"):
        g = g.sort_values("s", ascending=False)
        rank = next((i + 1 for i, (_, r) in enumerate(g.iterrows())
                     if r["cand"] == r["truth"]), 10 ** 9)
        rr.append(1 / rank if rank <= 25 else 0.0)
    return float(np.mean(rr))


def evaluate(feat, cols, params):
    out = []
    for held in SEEDS:
        va = feat[feat["seed"] == held]
        trn = feat[feat["seed"] != held]
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(trn[cols], trn["y"])
        v = va.copy()
        v["s"] = clf.predict_proba(v[cols])[:, 1]
        out.append(mrr_of(v))
    return out


def main():
    import pickle
    feats = ["cos", "ent", "ana"]
    feat = pd.concat([pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{s}.parquet")
                      for s in SEEDS], ignore_index=True)
    feat["y"] = (feat["cand"] == feat["truth"]).astype(int)
    grid = {
        "mine-made-up": dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                             l2_regularization=10.0),
        "theirs": dict(max_depth=6, max_iter=500, learning_rate=0.03,
                       min_samples_leaf=80, l2_regularization=1.0),
    }
    for lr, leaves, l2 in itertools.product((0.03, 0.05, 0.1), (7, 15, 31), (1.0, 10.0)):
        grid[f"lr{lr}-lf{leaves}-l2{l2}"] = dict(
            max_iter=300, learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2)
    print(f"{'config':22s} " + " ".join(f"s{s}" for s in SEEDS) + " mean", flush=True)
    rows = {}
    for name, params in grid.items():
        ms = evaluate(feat, feats, params)
        rows[name] = ms
        print(f"{name:22s} " + " ".join(f"{v:.3f}" for v in ms) +
              f" {np.mean(ms):.3f}", flush=True)
    with open(f"{PROJECT}/v13/gbm_grid.pkl", "wb") as f:
        pickle.dump(rows, f)


if __name__ == "__main__":
    main()
