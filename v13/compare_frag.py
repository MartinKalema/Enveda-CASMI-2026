"""Head-to-head: fork frag_scores vs our BDE frag, identical frozen queries."""
import numpy as np, pandas as pd
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v13.frag_up import frag_score_bde
import v13.fork_frag_raw as FF

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def nm(p, a):
    if a == "[2M+H]+": return (p - 1.007276) / 2
    if a == "[2M+Na]+": return (p - 22.989218) / 2
    if a == "[2M-H]-": return (p + 1.007276) / 2
    d = ADDUCT_DELTA.get(a)
    return p - d if d is not None else np.nan


def run(n_query=30, seed=30):
    rng = np.random.default_rng(seed)
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities"])
    tr["neutral"] = [nm(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    tr = tr[np.isfinite(tr["neutral"].values)]
    groups = np.array(tr["inchikey14"].unique())
    held = set(rng.choice(groups, size=500, replace=False))
    qp = tr[tr["inchikey14"].isin(held)]
    structs = np.array(qp["normalized_smiles"].unique())
    rng.shuffle(structs)
    queries = list(structs[:n_query])
    db = tr[~tr["inchikey14"].isin(held)]
    st = db.groupby("normalized_smiles")["neutral"].median().reset_index()
    tq = qp.groupby("normalized_smiles")["neutral"].median().reset_index()
    st = pd.concat([st, tq]).drop_duplicates("normalized_smiles").sort_values("neutral").reset_index(drop=True)
    masses = st["neutral"].values
    smi = st["normalized_smiles"].values
    a1, a2, cor = [], [], []
    for qi, qs in enumerate(queries):
        qspec = qp[qp["normalized_smiles"] == qs]
        q = qspec.iloc[0]
        qmass = float(np.median([nm(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        tol = qmass * 20 / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        cands = list(smi[lo:hi][:300])
        specs = [(q["ms2_mzs"], q["ms2_normalized_intensities"])]
        theirs = FF.frag_scores(cands, specs, 1.0, workers=1)
        s1 = [(float(v), s) for v, s in zip(theirs, cands)]
        s2 = []
        for s in cands:
            a, b = frag_score_bde(q["ms2_mzs"], q["ms2_normalized_intensities"],
                                  s, q["adduct"], q["precursor_mz"])
            s2.append((a + 2 * b, s))
        s1.sort(reverse=True); s2.sort(reverse=True)
        r1 = next((i + 1 for i, (_, s) in enumerate(s1) if s == qs), 10**9)
        r2 = next((i + 1 for i, (_, s) in enumerate(s2) if s == qs), 10**9)
        a1.append(1 / r1 if r1 <= 25 else 0.0)
        a2.append(1 / r2 if r2 <= 25 else 0.0)
        x = np.array([v for v, _ in s1]); y = np.array([v for v, _ in s2])
        if x.std() > 0 and y.std() > 0:
            cor.append(float(np.corrcoef(x, y)[0, 1]))
        if (qi + 1) % 10 == 0:
            print(f"{qi+1}/{n_query} theirs={np.mean(a1):.3f} ours={np.mean(a2):.3f}", flush=True)
    print(f"THEIRS MRR={np.mean(a1):.3f} OURS MRR={np.mean(a2):.3f} corr={np.mean(cor):.3f}")


if __name__ == "__main__":
    run()
