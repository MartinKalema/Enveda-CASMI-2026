"""v7 validation: retrieval-style (spectra held out, truth in pool) + disjoint.
Compares floor variants: cosine@20ppm (anchor) vs cosine@10ppm vs
entropy@10ppm vs formula-mass index. n=100 retrieval, n=50 disjoint.
"""
import numpy as np, pandas as pd
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine
from v4.channels import entropy_similarity
from v7.floor import formula_exact_mass, neutral_from_precursor, hybrid_tol

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def run(n_ret=100, n_dis=50, seed=20):
    rng = np.random.default_rng(seed)
    train = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "molecular_formula", "adduct",
                 "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_from_precursor(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    # formula-mass index (exact mass from formula; fallback precursor median)
    fmass = {}
    for s, f in zip(train["normalized_smiles"], train["molecular_formula"]):
        if s not in fmass:
            m = formula_exact_mass(f)
            if m is not None: fmass[s] = m
    struct = train.groupby("normalized_smiles")["neutral"].median().reset_index()
    struct["fmass"] = struct["normalized_smiles"].map(fmass)
    struct["fmass"] = struct["fmass"].fillna(struct["neutral"])
    sp = train.groupby("normalized_smiles").head(2).reset_index(drop=True)

    def cands(qmass, col="neutral", ppm=10.0, cap=3000):
        m = struct.sort_values(col)
        arr, s = m[col].values, m["normalized_smiles"].values
        tol = hybrid_tol(qmass, ppm) if col == "fmass" else qmass * ppm / 1e6
        lo = bisect_left(arr, qmass - tol); hi = bisect_right(arr, qmass + tol)
        return list(s[lo:hi][:cap])

    def score(qrow, cand_list, fn):
        out = {}
        for _, t in sp[sp["normalized_smiles"].isin(set(cand_list))].iterrows():
            v = fn(qrow["ms2_mzs"], qrow["ms2_normalized_intensities"],
                   t["ms2_mzs"], t["ms2_normalized_intensities"])
            k = t["normalized_smiles"]
            if v > out.get(k, 0): out[k] = v
        return out

    # retrieval-style: hide 1 spectrum per structure, truth stays pooled
    multi = train.groupby("normalized_smiles").filter(lambda g: len(g) >= 2)
    pool_structs = np.array(multi["normalized_smiles"].unique())
    rng.shuffle(pool_structs)
    variants = {"cos20": [], "cos10": [], "ent10": [], "ent10f": []}
    for qs in pool_structs[:n_ret]:
        g = multi[multi["normalized_smiles"] == qs]
        q = g.iloc[0]
        qmass = float(neutral_from_precursor(q["precursor_mz"], q["adduct"]))
        c20 = cands(qmass, "neutral", 20.0)
        c10 = cands(qmass, "neutral", 10.0)
        c10f = cands(qmass, "fmass", 10.0)
        s_cos20 = score(q, c20, cosine)
        s_cos10 = score(q, c10, cosine)
        s_ent10 = score(q, c10, entropy_similarity)
        s_ent10f = score(q, c10f, entropy_similarity)
        for name, sc, cl in [("cos20", s_cos20, c20), ("cos10", s_cos10, c10),
                             ("ent10", s_ent10, c10), ("ent10f", s_ent10f, c10f)]:
            r = sorted(((v, s) for s, v in sc.items()), reverse=True)
            rank = next((i + 1 for i, (_, s) in enumerate(r) if s == qs), 10**9)
            variants[name].append(1 / rank if rank <= 25 else 0.0)
    print("RETRIEVAL (truth pooled):", flush=True)
    for k, v in variants.items():
        print(f"  {k}: MRR25={np.mean(v):.3f}", flush=True)

    # disjoint: structures fully held out (novel regime)
    groups = np.array(train["inchikey14"].unique())
    held = set(rng.choice(groups, size=min(800, len(groups) // 12), replace=False))
    qpool = train[train["inchikey14"].isin(held)]
    structs = np.array(qpool["normalized_smiles"].unique())
    rng.shuffle(structs)
    dis = {"cos10": [], "ent10f": []}
    for qs in structs[:n_dis]:
        qspec = qpool[qpool["normalized_smiles"] == qs]
        q = qspec.iloc[0]
        qmass = float(np.median([neutral_from_precursor(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        for name, col in [("cos10", "neutral"), ("ent10f", "fmass")]:
            cl = cands(qmass, col, 10.0)
            fn = cosine if name == "cos10" else entropy_similarity
            sc = score(q, cl, fn)
            r = sorted(((v, s) for s, v in sc.items()), reverse=True)
            rank = next((i + 1 for i, (_, s) in enumerate(r) if s == qs), 10**9)
            dis[name].append(1 / rank if rank <= 25 else 0.0)
    print("DISJOINT (novels):", flush=True)
    for k, v in dis.items():
        print(f"  {k}: MRR25={np.mean(v):.3f}", flush=True)


if __name__ == "__main__":
    run()
