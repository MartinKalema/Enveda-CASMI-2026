"""Enrich cached seeds with BDE-ladder scores; train GBM variants.
Compares: gbm-oldfrag vs gbm-bde vs gbm-both. Saves winner to v13/gbm_bde.pkl
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier

from v13.frag_up import cached_fragments_bde

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)
BASE = ["cos", "ent", "ana", "mass_err", "log_prior", "t_top1", "metfrag", "fpdot"]
H = 1.007825


def ladder_score(mz, it, frags, top_n=100, tol=0.01):
    frags = np.asarray(sorted(frags), dtype=float)
    if len(frags) == 0:
        return 0.0
    mz = np.asarray(mz, dtype=float)
    it = np.asarray(it, dtype=float)
    o = np.argsort(-it)[:top_n]
    mz, it = mz[o], it[o]
    w = np.sqrt(np.maximum(it, 0))
    tot = w.sum()
    if tot <= 0:
        return 0.0
    ion = np.sort(np.concatenate([frags + dh * H + 1.007276 for dh in (-2, -1, 0, 1, 2)]))
    idx = np.searchsorted(ion, mz)
    ok = np.zeros(len(mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - mz) <= tol
    return float(w[ok].sum() / tot)


def main():
    tr = pd.read_parquet(
        f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "adduct", "ms2_mzs", "ms2_normalized_intensities"])
    tr = tr.set_index("normalized_smiles", append=False)
    parts = []
    for seed in SEEDS:
        d = pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{seed}.parquet")
        try:
            extra = pd.read_parquet(f"{PROJECT}/v13/feat_bde_seed{seed}.parquet")
            d = d.copy()
            d["bde"] = extra["bde"].values
            parts.append(d)
            print(f"seed {seed}: cache hit", flush=True)
            continue
        except FileNotFoundError:
            pass
        d = d.copy()
        vals = []
        for (sd, truth), g in d.groupby(["seed", "truth"]):
            try:
                rows = tr.loc[[truth]].head(2)
            except KeyError:
                vals.extend([0.0] * len(g))
                continue
            if len(rows) == 0:
                vals.extend([0.0] * len(g))
                continue
            best = {}
            for _, r in rows.iterrows():
                for _, c in g.iterrows():
                    f = cached_fragments_bde(c["cand"], r["adduct"])
                    v = ladder_score(r["ms2_mzs"], r["ms2_normalized_intensities"], f)
                    if v > best.get(c["cand"], 0.0):
                        best[c["cand"]] = v
            vals.extend(best.get(c["cand"], 0.0) for _, c in g.iterrows())
        d["bde"] = vals
        d[["bde"]].to_parquet(f"{PROJECT}/v13/feat_bde_seed{seed}.parquet")
        parts.append(d)
        print(f"seed {seed}: computed", flush=True)
    feat = pd.concat(parts, ignore_index=True)
    feat["y"] = (feat["cand"] == feat["truth"]).astype(int)
    feat[["bde"]] = feat[["bde"]].fillna(0)
    print("rows:", len(feat), "pos:", feat["y"].sum(), flush=True)
    variants = {
        "gbm-oldfrag": BASE,
        "gbm-bde": [c for c in BASE if c != "metfrag"] + ["bde"],
        "gbm-both": BASE + ["bde"],
    }
    for held in SEEDS:
        va = feat[feat["seed"] == held]
        trn = feat[feat["seed"] != held]
        line = f"held={held}"
        for name, cols in variants.items():
            clf = HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.05,
                                                 max_leaf_nodes=15, l2_regularization=10.0)
            clf.fit(trn[cols], trn["y"])
            p = clf.predict_proba(va[cols])[:, 1]
            v = va.copy()
            v["s"] = p
            rr = []
            for _, gg in v.groupby("truth"):
                gg = gg.sort_values("s", ascending=False)
                rank = next((i + 1 for i, (_, r) in enumerate(gg.iterrows())
                             if r["cand"] == r["truth"]), 10 ** 9)
                rr.append(1 / rank if rank <= 25 else 0.0)
            line += f" {name}={np.mean(rr):.3f}"
        print(line, flush=True)
    final = HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=15, l2_regularization=10.0)
    final.fit(feat[[c for c in BASE if c != "metfrag"] + ["bde"]], feat["y"])
    with open(f"{PROJECT}/v13/gbm_bde.pkl", "wb") as f:
        pickle.dump({"model": final,
                     "feats": [c for c in BASE if c != "metfrag"] + ["bde"]}, f)
    print("saved v13/gbm_bde.pkl", flush=True)


if __name__ == "__main__":
    main()
