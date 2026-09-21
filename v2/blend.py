"""v2: memory-based fingerprint prediction (MIST retrieval path, no training).
Query spectra -> cosine neighbors among mass-window train spectra ->
cosine-weighted fingerprint blend -> rank candidates by Tanimoto.
"""
import numpy as np, pandas as pd
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def cosine(mz1, it1, mz2, it2, tol=0.02):
    a = np.asarray(it1, dtype=float); b = np.asarray(it2, dtype=float)
    na = float(np.sqrt((a * a).sum())); nb = float(np.sqrt((b * b).sum()))
    if na == 0 or nb == 0: return 0.0
    m1 = np.asarray(mz1, dtype=float); m2 = np.asarray(mz2, dtype=float)
    o1 = np.argsort(m1); o2 = np.argsort(m2)
    m1, a = m1[o1], a[o1]; m2, b = m2[o2], b[o2]
    i = j = 0; num = 0.0
    while i < len(m1) and j < len(m2):
        d = m1[i] - m2[j]
        if abs(d) <= tol: num += a[i] * b[j]; i += 1; j += 1
        elif d < 0: i += 1
        else: j += 1
    return num / (na * nb)


def tanimoto(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def load_fp():
    df = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    return {s: np.unpackbits(np.asarray(f, dtype=np.uint8)) for s, f in zip(df["smiles"], df["fp"])}


def run(n_query=30, seed=1, k_blend=15, ppm=20, min_n=200):
    rng = np.random.default_rng(seed)
    fp = load_fp()
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
    # sampled db spectra per structure for neighbor search
    db_samp = db.groupby("normalized_smiles").head(3)

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
        cands = list(smi[lo:hi][:2000])
        win = db_samp[db_samp["normalized_smiles"].isin(set(cands))]
        # neighbor similarities: best cosine of each db spectrum vs query spectra
        sims = []
        for _, r in qspec.iterrows():
            for _, t in win.iterrows():
                c = cosine(r["ms2_mzs"], r["ms2_normalized_intensities"],
                           t["ms2_mzs"], t["ms2_normalized_intensities"])
                if c > 0.01:
                    sims.append((c, t["normalized_smiles"]))
        sims.sort(reverse=True)
        # blend top-k distinct neighbor fingerprints
        seen, num, den = set(), None, 0.0
        for c, s in sims:
            if s in seen: continue
            seen.add(s)
            f = fp.get(s)
            if f is None: continue
            num = c * f if num is None else num + c * f
            den += c
            if len(seen) >= k_blend: break
        pred = (num / den) if den > 0 else None
        scored = []
        for s in cands:
            f = fp.get(s)
            if f is None or pred is None: t = 0.0
            else: t = tanimoto(pred > 0.3, f)
            scored.append((t, s))
        scored.sort(reverse=True)
        rank = next((i + 1 for i, (_, s) in enumerate(scored) if s == qs), 10**9)
        rr.append(1 / rank if rank <= 25 else 0.0)
        for k in hits:
            if rank <= k: hits[k] += 1
        if (qi + 1) % 10 == 0: print(f"{qi+1}/{n_query} MRR25={np.mean(rr):.3f}", flush=True)
    print(f"n={n_query} hit@1={hits[1]/n_query:.3f} hit@5={hits[5]/n_query:.3f} hit@25={hits[25]/n_query:.3f} MRR@25={np.mean(rr):.3f}")


if __name__ == "__main__":
    run()
