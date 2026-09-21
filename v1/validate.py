"""v1 validation: structure-disjoint retrieval, hit@k (MRR proxy).
Hold out inchikey14 groups; query = their spectra; candidates = mass window
over REMAINING structures (truth injected via mass window like production).
Score = max explained-intensity over query spectra (formula-labelled).
"""
import numpy as np, pandas as pd
from bisect import bisect_left, bisect_right
from .subformula import explained_intensity, ADDUCT_DELTA

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def run(n_query=50, seed=0, ppm=20, min_n=200):
    rng = np.random.default_rng(seed)
    train = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "molecular_formula",
                 "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    groups = np.array(train["inchikey14"].unique())
    held = set(rng.choice(groups, size=min(500, len(groups) // 20), replace=False))
    qpool = train[train["inchikey14"].isin(held)]
    # pick n_query structures with >=1 spectrum
    structs = np.array(qpool["normalized_smiles"].unique())
    rng.shuffle(structs)
    queries = list(structs[:n_query])

    db = train[~train["inchikey14"].isin(held)]
    struct = db.groupby("normalized_smiles").agg(
        mass=("neutral", "median"), formula=("molecular_formula", "first")).reset_index()
    # inject truth structures (mass/formula only, spectra stay hidden)
    truth = qpool.groupby("normalized_smiles").agg(
        mass=("neutral", "median"), formula=("molecular_formula", "first")).reset_index()
    struct = pd.concat([struct, truth]).drop_duplicates("normalized_smiles")
    struct = struct.sort_values("mass").reset_index(drop=True)
    masses = struct["mass"].values
    smi = struct["normalized_smiles"].values
    fmap = dict(zip(struct["normalized_smiles"], struct["formula"]))

    cache = {}
    def expl(mz, it, formula, adduct):
        key = (id(mz), formula, adduct)
        if key not in cache:
            cache[key] = explained_intensity(mz, it, formula, adduct)[0]
        return cache[key]

    hits = {1: 0, 5: 0, 25: 0}
    rr = []
    for qi, qs in enumerate(queries):
        qspec = qpool[qpool["normalized_smiles"] == qs]
        qmass = float(np.median([neutral_mass(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        tol = qmass * ppm / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        pcur = ppm
        while hi - lo < min_n and pcur < 500:
            pcur *= 2; tol = qmass * pcur / 1e6
            lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        cands = smi[lo:hi][:2000]
        scored = []
        for s in cands:
            f = fmap[s]
            best = 0.0
            for _, r in qspec.iterrows():
                v = expl(r["ms2_mzs"], r["ms2_normalized_intensities"], f, r["adduct"])
                if v > best: best = v
            scored.append((best, s))
        scored.sort(reverse=True)
        rank = next((i + 1 for i, (_, s) in enumerate(scored) if s == qs), None)
        if rank is None: rank = 10**9
        rr.append(1 / rank if rank <= 25 else 0.0)
        for k in hits:
            if rank <= k: hits[k] += 1
        if (qi + 1) % 10 == 0: print(f"{qi+1}/{n_query} MRR25={np.mean(rr):.3f}", flush=True)
    print(f"n={n_query} hit@1={hits[1]/n_query:.3f} hit@5={hits[5]/n_query:.3f} hit@25={hits[25]/n_query:.3f} MRR@25={np.mean(rr):.3f}")


if __name__ == "__main__":
    run(n_query=30)
