"""GBM-25: expand cached channels with mass/formula/fingerprint/fragment features.
Trains on 2 seeds, evaluates on held-out 3rd. No submission needed.
"""
import numpy as np, pandas as pd, pickle
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)


def tanimoto(a, b):
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def main():
    feat = pd.concat([pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{s}.parquet") for s in SEEDS],
                     ignore_index=True)
    feat["y"] = (feat["cand"] == feat["truth"]).astype(int)
    print("rows:", len(feat), "pos:", feat["y"].sum(), flush=True)
    # candidate masses + formulae
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "molecular_formula", "precursor_mz", "adduct",
                 "ms2_mzs", "ms2_normalized_intensities"])
    from v1.subformula import ADDUCT_DELTA

    def nm(p, a):
        if a == "[2M+H]+": return (p - 1.007276) / 2
        if a == "[2M+Na]+": return (p - 22.989218) / 2
        if a == "[2M-H]-": return (p + 1.007276) / 2
        d = ADDUCT_DELTA.get(a)
        return p - d if d is not None else np.nan
    tr["neutral"] = [nm(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    smass = tr.groupby("normalized_smiles")["neutral"].median().to_dict()
    sform = tr.groupby("normalized_smiles")["molecular_formula"].first().to_dict()
    cf = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet",
        columns=["canonical_smiles", "molecular_formula", "exact_molecular_weight"])
    for s, m, f in zip(cf["canonical_smiles"], cf["exact_molecular_weight"], cf["molecular_formula"]):
        smass.setdefault(s, float(m)); sform.setdefault(s, f)
    from collections import Counter
    fprior = Counter(tr["molecular_formula"].tolist())
    # fingerprints for tanimoto-to-top1
    tf = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    scol = "normalized_smiles" if "normalized_smiles" in tf.columns else "smiles"
    fp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8)) for s, f in zip(tf[scol], tf["fp"])}
    cf2 = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet", columns=["canonical_smiles", "fp"])
    for s, f in zip(cf2["canonical_smiles"], cf2["fp"]):
        fp.setdefault(s, np.unpackbits(np.asarray(f, dtype=np.uint8)))
    with open(f"{PROJECT}/data/frag_cache.pkl", "rb") as fh:
        frags = pickle.load(fh)
    # query masses: median neutral per (seed, truth) from candidate truth rows is unknown;
    # approximate with candidate mass of truth row itself
    feat["cmass"] = feat["cand"].map(smass)
    feat["qmass"] = feat.groupby(["seed", "truth"])["cmass"].transform("median")
    feat["mass_err"] = (feat["cmass"] - feat["qmass"]).abs() / feat["qmass"]
    feat["log_prior"] = feat["cand"].map(lambda s: np.log1p(fprior.get(sform.get(s, ""), 0)))
    # tanimoto to channel-top1 per query
    feat["t_top1"] = 0.0
    for (seed, truth), g in feat.groupby(["seed", "truth"]):
        top = g.sort_values("ana", ascending=False)["cand"].head(3).tolist()
        tv = [fp[t] for t in top if t in fp]
        if not tv: continue
        for i in g.index:
            f = fp.get(feat.at[i, "cand"])
            if f is not None:
                feat.at[i, "t_top1"] = max(tanimoto(f, t) for t in tv)
    # frag score needs spectra - expensive; use cache hit count proxy: skip, use mass/formula only
    FE = ["cos", "ent", "ana", "mass_err", "log_prior", "t_top1"]
    feat[FE] = feat[FE].fillna(0)
    for held in SEEDS:
        trn = feat[feat["seed"] != held]; va = feat[feat["seed"] == held]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=15, l2_regularization=10.0)
        clf.fit(trn[FE], trn["y"])
        for name, col in [("gbm25", None), ("gbm3", ["cos", "ent", "ana"]), ("ana", None)]:
            if name == "gbm25":
                va = va.copy(); va["s"] = clf.predict_proba(va[FE])[:, 1]
            elif name == "gbm3":
                c3 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                    max_leaf_nodes=15, l2_regularization=10.0)
                c3.fit(trn[["cos", "ent", "ana"]], trn["y"])
                va = va.copy(); va["s"] = c3.predict_proba(va[["cos", "ent", "ana"]])[:, 1]
            else:
                va = va.copy(); va["s"] = va["ana"]
            rr = []
            for _, g in va.groupby("truth"):
                g = g.sort_values("s", ascending=False)
                rank = next((i + 1 for i, (_, r) in enumerate(g.iterrows()) if r["cand"] == r["truth"]), 10**9)
                rr.append(1 / rank if rank <= 25 else 0.0)
            print(f"held={held} {name}={np.mean(rr):.3f}", flush=True)


if __name__ == "__main__":
    main()
