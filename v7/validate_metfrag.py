"""MetFrag-lite channel validation (disjoint, n=30): MRR + corr vs cosine/analog."""
import numpy as np, pandas as pd
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine
from v7.metfrag import frag_score, cached_fragments

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def run(n_query=30, seed=30):
    rng = np.random.default_rng(seed)
    train = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    groups = np.array(train["inchikey14"].unique())
    held = set(rng.choice(groups, size=min(500, len(groups) // 20), replace=False))
    qpool = train[train["inchikey14"].isin(held)]
    structs = np.array(qpool["normalized_smiles"].unique())
    rng.shuffle(structs)
    queries = list(structs[:n_query])
    db = train[~train["inchikey14"].isin(held)]
    struct = db.groupby("normalized_smiles")["neutral"].median().reset_index()
    truth = qpool.groupby("normalized_smiles")["neutral"].median().reset_index()
    struct = pd.concat([struct, truth]).drop_duplicates("normalized_smiles")
    struct = struct.sort_values("neutral").reset_index(drop=True)
    masses = struct["neutral"].values
    smi = struct["normalized_smiles"].values
    db_samp = db.groupby("normalized_smiles").head(2)

    mrr_m = mrr_c = []
    mrr_m, mrr_c = [], []
    corrs = []
    for qi, qs in enumerate(queries):
        qspec = qpool[qpool["normalized_smiles"] == qs]
        q = qspec.iloc[0]
        qmass = float(np.median([neutral_mass(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        tol = qmass * 20 / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        cands = list(smi[lo:hi][:300])
        for s in cands: cached_fragments(s)  # warm
        win = db_samp[db_samp["normalized_smiles"].isin(set(cands))]
        sm, sc = [], []
        for s in cands:
            f = frag_score(q["ms2_mzs"], q["ms2_normalized_intensities"], s, q["adduct"])
            best = 0.0
            for _, t in win[win["normalized_smiles"] == s].iterrows():
                c = cosine(q["ms2_mzs"], q["ms2_normalized_intensities"],
                           t["ms2_mzs"], t["ms2_normalized_intensities"])
                if c > best: best = c
            sm.append((f, s)); sc.append((best, s))
        sm.sort(reverse=True); sc.sort(reverse=True)
        rm = next((i + 1 for i, (_, s) in enumerate(sm) if s == qs), 10**9)
        rc = next((i + 1 for i, (_, s) in enumerate(sc) if s == qs), 10**9)
        mrr_m.append(1 / rm if rm <= 25 else 0.0)
        mrr_c.append(1 / rc if rc <= 25 else 0.0)
        dm = {s: v for v, s in sm}; dc = {s: v for v, s in sc}
        x = np.array([dm[s] for s in cands]); y = np.array([dc[s] for s in cands])
        if x.std() > 0 and y.std() > 0:
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
        if (qi + 1) % 10 == 0: print(f"{qi+1}/{n_query}", flush=True)
    print(f"metfrag MRR={np.mean(mrr_m):.3f} cosine MRR={np.mean(mrr_c):.3f} corr={np.mean(corrs):.3f}")


if __name__ == "__main__":
    run()
