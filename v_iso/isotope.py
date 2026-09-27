"""Isotope satellite rerank prototype (SOTA N1).
For each query spectrum: find M+1 (+1.0034 Da) and M+2 (+2.0057) satellites
of top peaks; observed satellite pattern vs formula-expected (C 1.08%, Br
M+2 98%, Cl M+2 32%, S M+2 4.2%). Score candidates by agreement.
Validate disjoint n=60: does isotope agreement rank truth above decoys?
"""
import numpy as np, pandas as pd
from v1.subformula import parse_formula

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
# natural abundances for M+1 / M+2 contribution per atom
M1 = {"C": 0.0108, "H": 0.000115, "N": 0.00364, "O": 0.00038, "S": 0.0076,
      "Cl": 0.0, "Br": 0.0, "F": 0.0, "P": 0.0, "I": 0.0}
M2 = {"C": 0.0, "H": 0.0, "N": 0.0, "O": 0.00205, "S": 0.0422,
      "Cl": 0.3199, "Br": 0.9782, "F": 0.0, "P": 0.0, "I": 0.0}


def expected_satellites(formula_str):
    f = parse_formula(formula_str)
    if f is None:
        return None
    m1 = sum(n * M1.get(e, 0.0) for e, n in f.items())
    m2 = sum(n * M2.get(e, 0.0) for e, n in f.items())
    return m1, m2


def observed_satellites(mzs, intens, tol=0.01, top_n=40):
    """Median observed M+1, M+2 ratios over top peaks having a satellite."""
    mz = np.asarray(mzs, dtype=float); it = np.asarray(intens, dtype=float)
    o = np.argsort(-it)[:top_n]
    mz, it = mz[o], it[o]
    r1, r2 = [], []
    for i in range(len(mz)):
        d1 = np.abs(mz - (mz[i] + 1.003355))
        j = int(np.argmin(d1))
        if d1[j] <= tol and it[i] > 0:
            r1.append(it[j] / it[i])
        d2 = np.abs(mz - (mz[i] + 2.0057))
        k = int(np.argmin(d2))
        if d2[k] <= tol and it[i] > 0:
            r2.append(it[k] / it[i])
    f1 = float(np.median(r1)) if r1 else 0.0
    f2 = float(np.median(r2)) if r2 else 0.0
    return f1, f2, len(r1), len(r2)


def iso_score(formula_str, obs1, obs2):
    exp = expected_satellites(formula_str)
    if exp is None:
        return 0.0
    e1, e2 = exp
    # agreement: negative abs log-ratio distance, Br/Cl M+2 dominates
    s = -abs(np.log1p(obs1) - np.log1p(e1)) - abs(np.log1p(obs2) - np.log1p(e2))
    return float(s)


def run(n_query=60, seed=50):
    from bisect import bisect_left, bisect_right
    from v1.subformula import ADDUCT_DELTA

    def nm(p, a):
        if a == "[2M+H]+": return (p - 1.007276) / 2
        if a == "[2M+Na]+": return (p - 22.989218) / 2
        if a == "[2M-H]-": return (p + 1.007276) / 2
        d = ADDUCT_DELTA.get(a)
        return p - d if d is not None else np.nan

    rng = np.random.default_rng(seed)
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "molecular_formula", "adduct",
                 "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    tr["neutral"] = [nm(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    tr = tr[np.isfinite(tr["neutral"].values)]
    groups = np.array(tr["inchikey14"].unique())
    held = set(rng.choice(groups, size=min(800, len(groups) // 12), replace=False))
    qp = tr[tr["inchikey14"].isin(held)]
    structs = np.array(qp["normalized_smiles"].unique())
    rng.shuffle(structs)
    db = tr[~tr["inchikey14"].isin(held)]
    st = db.groupby("normalized_smiles").agg(
        mass=("neutral", "median"), formula=("molecular_formula", "first")).reset_index()
    truth = qp.groupby("normalized_smiles").agg(
        mass=("neutral", "median"), formula=("molecular_formula", "first")).reset_index()
    st = pd.concat([st, truth]).drop_duplicates("normalized_smiles").sort_values("mass").reset_index(drop=True)
    masses = st["mass"].values
    smi = st["normalized_smiles"].values
    fmap = dict(zip(st["normalized_smiles"], st["formula"]))
    rr = []
    for qi, qs in enumerate(structs[:n_query]):
        qspec = qp[qp["normalized_smiles"] == qs]
        qmass = float(np.median([nm(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        tol = qmass * 20 / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        cands = list(smi[lo:hi][:800])
        # observed satellites: max over query spectra
        o1 = o2 = 0.0
        for _, r in qspec.iterrows():
            a1, a2, _, _ = observed_satellites(r["ms2_mzs"], r["ms2_normalized_intensities"])
            o1 = max(o1, a1); o2 = max(o2, a2)
        scored = [(iso_score(fmap.get(s, ""), o1, o2), s) for s in cands]
        scored.sort(reverse=True)
        rank = next((i + 1 for i, (_, s) in enumerate(scored) if s == qs), 10**9)
        rr.append(1 / rank if rank <= 25 else 0.0)
        if (qi + 1) % 15 == 0:
            print(f"{qi+1}/{n_query} iso-MRR={np.mean(rr):.3f}", flush=True)
    print(f"isotope-rerank MRR25={np.mean(rr):.3f} (n={len(rr)})")


if __name__ == "__main__":
    run()
