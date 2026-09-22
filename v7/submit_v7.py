"""v7 production: cosine top-5 floor + MetFrag-ranked fills (precomputed frags).
Writes submission.csv. Needs data/frag_cache.pkl (ships via fp dataset).
RDKit needed ONLY for precompute, never at inference (numpy matching).
"""
import numpy as np, pandas as pd, os, pickle
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")
OUT = os.environ.get("CASMI_OUT", PROJECT)
FP = os.environ.get("CASMI_FP", f"{PROJECT}/data")
TOP_N = 150
FRAG_TOL = 0.01

AD = {"[M+H]+": 1.007276, "[M+Na]+": 22.989218, "[M+K]+": 38.963158,
      "[M+NH4]+": 18.033823, "[M-H]-": -1.007276, "[M+Cl]-": 34.968853,
      "[M+CH2O2-H]-": 44.998201, "[M+C2H4O2-H]-": 59.013851, "[M]+": 0.0}


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def topn(mz, it, n=TOP_N):
    mz = np.asarray(mz, dtype=float); it = np.asarray(it, dtype=float)
    if len(mz) <= n: return mz, it
    o = np.argsort(-it)[:n]
    return mz[o], it[o]


def window(masses, smi, qmass, ppm=20, min_n=200, cap=2000):
    tol = qmass * ppm / 1e6
    lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
    pcur = ppm
    while hi - lo < min_n and pcur < 500:
        pcur *= 2; tol = qmass * pcur / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
    return list(smi[lo:hi][:cap])


def frag_match(mz, it, frags, adduct, tol=FRAG_TOL, top_n=100):
    """sqrt-intensity fraction explained using precomputed fragment masses."""
    if frags is None or len(frags) == 0: return 0.0
    d = AD.get(adduct, 1.007276)
    mz = np.asarray(mz, dtype=float); it = np.asarray(it, dtype=float)
    o = np.argsort(-it)[:top_n]
    mz, it = mz[o], it[o]
    w = np.sqrt(np.maximum(it, 0))
    tot = w.sum()
    if tot <= 0: return 0.0
    frags = np.asarray(frags, dtype=float)
    hit = 0.0
    for m, wi in zip(mz, w):
        t = m - d
        j = int(np.searchsorted(frags, t))
        for jj in (j - 1, j):
            if 0 <= jj < len(frags) and abs(frags[jj] - t) <= tol:
                hit += wi; break
    return float(hit / tot)


def main():
    with open(f"{FP}/frag_cache.pkl", "rb") as f:
        frags = pickle.load(f)
    print(f"frag sets: {len(frags)}", flush=True)
    test = pd.read_parquet(f"{IN}/test.parquet")
    test["neutral"] = [neutral_mass(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()
    train = pd.read_parquet(f"{IN}/train.parquet",
        columns=["normalized_smiles", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    tstruct = train.groupby("normalized_smiles")["neutral"].median()
    tmass = tstruct.sort_values().values
    tsmi = tstruct.sort_values().index.values
    tsamp = train.groupby("normalized_smiles").head(2)
    cf = pd.read_parquet(f"{FP}/coconut_fp.parquet", columns=["canonical_smiles", "exact_molecular_weight"])
    co = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values

    rows = []
    for mi, (mol, spectra) in enumerate(test.groupby("molecule_id")):
        qmass = float(mol_neutral.loc[mol])
        tcands = window(tmass, tsmi, qmass)
        twin = tsamp[tsamp["normalized_smiles"].isin(set(tcands))]
        qspecs = [topn(np.asarray(mz, dtype=float), np.asarray(it, dtype=float))
                  for mz, it in zip(spectra["ms2_mzs"], spectra["ms2_normalized_intensities"])]
        tscored = []
        for s in tcands:
            best = 0.0
            for tmz, tit in zip(twin[twin["normalized_smiles"] == s]["ms2_mzs"],
                               twin[twin["normalized_smiles"] == s]["ms2_normalized_intensities"]):
                dmz, dit = topn(tmz, tit)
                for qmz, qit in qspecs:
                    c = cosine(qmz, qit, dmz, dit)
                    if c > best: best = c
            tscored.append((best, s))
        tscored.sort(reverse=True)
        top5 = [s for _, s in tscored[:5]]
        ccands = window(cmass, csmi, qmass, cap=3000)
        fills = []
        for s in tcands[len(top5):] + ccands:
            f = frags.get(s)
            if f is None: continue
            best = 0.0
            for _, r in spectra.iterrows():
                v = frag_match(r["ms2_mzs"], r["ms2_normalized_intensities"], f, r["adduct"])
                if v > best: best = v
            fills.append((best, s))
        fills.sort(reverse=True)
        seen = set(top5)
        out = list(top5) + [s for _, s in fills if not (s in seen or seen.add(s))][:20]
        rows.append((mol, ";".join(out[:25])))
        if (mi + 1) % 50 == 0: print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
