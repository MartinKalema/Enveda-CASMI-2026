"""GBM rerank over cached channel features (survey finding 7).
Train on 2 seeds, evaluate on held-out 3rd. Tests calibration > hand weights.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)
FEATS = ["cos", "ent", "ana"]


def load():
    return pd.concat([pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{s}.parquet") for s in SEEDS])


def mrr(d, col="s"):
    rr = []
    for truth, g in d.groupby("truth"):
        g = g.sort_values(col, ascending=False)
        rank = next((i + 1 for i, (_, r) in enumerate(g.iterrows()) if r["cand"] == truth), 10**9)
        rr.append(1 / rank if rank <= 25 else 0.0)
    return float(np.mean(rr))


def main():
    feat = load()
    feat["y"] = (feat["cand"] == feat["truth"]).astype(int)
    print("positives:", feat["y"].sum(), "/", len(feat), flush=True)
    for held in SEEDS:
        tr = feat[feat["seed"] != held]
        va = feat[feat["seed"] == held]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=15, l2_regularization=10.0)
        clf.fit(tr[FEATS], tr["y"])
        va = va.copy()
        va["gbm"] = clf.predict_proba(va[FEATS])[:, 1]
        va["hand"] = va[["cos", "ent", "ana"]].max(axis=1)
        print(f"held={held} gbm={mrr(va, 'gbm'):.3f} hand-max={mrr(va, 'hand'):.3f} "
              f"cos={mrr(va, 'cos'):.3f} ana={mrr(va, 'ana'):.3f}", flush=True)
    import pickle
    tr_all = feat
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=15, l2_regularization=10.0)
    clf.fit(tr_all[FEATS], tr_all["y"])
    with open(f"{PROJECT}/v5/gbm_ranker.pkl", "wb") as f:
        pickle.dump({"model": clf, "feats": FEATS}, f)
    print("saved v5/gbm_ranker.pkl", flush=True)


if __name__ == "__main__":
    main()
