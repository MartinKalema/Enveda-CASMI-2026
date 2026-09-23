"""Enrich v5 feat_cache with metfrag + fpdot, train bagged GBM. Saves v11/gbm_full.pkl
Usage: .venv/bin/python -m v11.enrich_train
"""
import numpy as np, pandas as pd, pickle, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from v7.submit_v7 import frag_match
from v2.train_fp import FpMLP

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)
FEATS = ["cos", "ent", "ana", "mass_err", "log_prior", "t_top1", "metfrag", "fpdot"]


def main():
    device = "cpu"
    model = FpMLP(d_h=1536).to(device)
    model.load_state_dict(torch.load(f"{PROJECT}/data/fp_trans.pt", map_location=device))
    model.eval()
    with open(f"{PROJECT}/data/frag_cache.pkl", "rb") as f:
        frags = pickle.load(f)
    tf = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    scol = "normalized_smiles" if "normalized_smiles" in tf.columns else "smiles"
    tfp = {s: np.unpackbits(np.asarray(x, dtype=np.uint8)).astype(np.float32)
           for s, x in zip(tf[scol], tf["fp"])}
    cf = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet", columns=["canonical_smiles", "fp"])
    for s, x in zip(cf["canonical_smiles"], cf["fp"]):
        tfp.setdefault(s, np.unpackbits(np.asarray(x, dtype=np.uint8)).astype(np.float32))
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "adduct", "precursor_mz", "ms2_mzs",
                 "ms2_normalized_intensities", "collision_energy_ev"])
    tr = tr.set_index("normalized_smiles", append=False)
    from v2.train_fp import bin_spectrum, meta_vec, N_BINS, ADDUCTS
    # per-query spectra cache for metfrag scoring
    parts = []
    for seed in SEEDS:
        d = pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{seed}.parquet")
        # metfrag needs query spectra: up to 2 spectra per truth from train
        d = d.copy()
        mf, fd = [], []
        for (sd, truth), g in d.groupby(["seed", "truth"]):
            try:
                rows = tr.loc[[truth]].head(2)
            except KeyError:
                rows = None
            if rows is None or len(rows) == 0:
                mf.extend([0.0] * len(g)); fd.extend([0.0] * len(g))
                continue
            # predicted fp (no truth leak): mean model logits over query spectra
            feats = []
            for _, r in rows.iterrows():
                v = np.zeros(N_BINS + len(ADDUCTS) + 2, dtype=np.float32)
                v[:N_BINS] = bin_spectrum(r["ms2_mzs"], r["ms2_normalized_intensities"])
                v[N_BINS:] = meta_vec(r["adduct"], r["precursor_mz"], r["collision_energy_ev"])
                feats.append(v)
            with torch.no_grad():
                Z = model(torch.from_numpy(np.stack(feats))).numpy().mean(axis=0)
            Zn = Z / (np.linalg.norm(Z) + 1e-9)
            for _, c in g.iterrows():
                best = 0.0
                for _, r in rows.iterrows():
                    v = frag_match(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                   frags.get(c["cand"], []), r["adduct"])
                    if v > best:
                        best = v
                mf.append(best)
                cf2 = tfp.get(c["cand"])
                fn = cf2 / (np.linalg.norm(cf2) + 1e-9) if cf2 is not None else None
                fd.append(float(fn @ Zn) if fn is not None else 0.0)
        d["metfrag"] = mf
        d["fpdot"] = fd
        parts.append(d)
    feat = pd.concat(parts, ignore_index=True)
    feat["y"] = (feat["cand"] == feat["truth"]).astype(int)
    # base features absent from cache: mass_err, log_prior, t_top1
    trb = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "molecular_formula", "precursor_mz", "adduct"])
    from v1.subformula import ADDUCT_DELTA as _AD

    def _nm(p, a):
        if a == "[2M+H]+": return (p - 1.007276) / 2
        if a == "[2M+Na]+": return (p - 22.989218) / 2
        if a == "[2M-H]-": return (p + 1.007276) / 2
        dd = _AD.get(a)
        return p - dd if dd is not None else np.nan
    trb["neutral"] = [_nm(p, a) for p, a in zip(trb["precursor_mz"], trb["adduct"])]
    _smas = trb.groupby("normalized_smiles")["neutral"].median().to_dict()
    _sform = trb.groupby("normalized_smiles")["molecular_formula"].first().to_dict()
    _cfm = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet",
        columns=["canonical_smiles", "molecular_formula", "exact_molecular_weight"])
    for s, m, f in zip(_cfm["canonical_smiles"], _cfm["exact_molecular_weight"], _cfm["molecular_formula"]):
        _smas.setdefault(s, float(m)); _sform.setdefault(s, f)
    from collections import Counter as _Counter
    _fprior = _Counter(trb["molecular_formula"].tolist())
    feat["cmass"] = feat["cand"].map(_smas)
    feat["qmass"] = feat.groupby(["seed", "truth"])["cmass"].transform("median")
    feat["mass_err"] = (feat["cmass"] - feat["qmass"]).abs() / feat["qmass"]
    feat["log_prior"] = feat["cand"].map(lambda s: float(np.log1p(_fprior.get(_sform.get(s, ""), 0))))
    feat["t_top1"] = 0.0
    for (seed, truth), g in feat.groupby(["seed", "truth"]):
        top = g.sort_values("ana", ascending=False)["cand"].head(3).tolist()
        tv = [tfp[t] for t in top if t in tfp]
        if not tv:
            continue
        for i in g.index:
            f = tfp.get(feat.at[i, "cand"])
            if f is not None:
                fb = f > 0.5
                feat.at[i, "t_top1"] = max(float(np.logical_and(fb, t > 0.5).sum()) /
                                          float(np.logical_or(fb, t > 0.5).sum()) for t in tv)
    feat[FEATS] = feat[FEATS].fillna(0)
    print("rows:", len(feat), "pos:", feat["y"].sum(), flush=True)
    for held in SEEDS:
        va = feat[feat["seed"] == held]
        trn = feat[feat["seed"] != held]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=15, l2_regularization=10.0)
        clf.fit(trn[FEATS], trn["y"])
        p = clf.predict_proba(va[FEATS])[:, 1]
        va = va.copy(); va["s"] = p
        rr = []
        for _, g in va.groupby("truth"):
            g = g.sort_values("s", ascending=False)
            rank = next((i + 1 for i, (_, r) in enumerate(g.iterrows()) if r["cand"] == r["truth"]), 10**9)
            rr.append(1 / rank if rank <= 25 else 0.0)
        print(f"held={held} gbm8={np.mean(rr):.3f}", flush=True)
    final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=15, l2_regularization=10.0)
    final.fit(feat[FEATS], feat["y"])
    with open(f"{PROJECT}/v11/gbm_full.pkl", "wb") as f:
        pickle.dump({"model": final, "feats": FEATS}, f)
    print("saved v11/gbm_full.pkl", flush=True)


if __name__ == "__main__":
    main()
