"""Baseline v0: mass-filter + spectral cosine, per-molecule aggregation.
Fast, no RDKit, runs on M4 CPU. Outputs submission.csv (400 x up to 25 SMILES).
"""
import numpy as np, pandas as pd, os
PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
TRAIN = f"{PROJECT}/data/train.parquet"
TEST = f"{PROJECT}/data/test.parquet"
OUT = f"{PROJECT}/submission_v0.csv"

# adduct -> delta such that precursor = M + delta (singly charged monomers)
ADDUCT_DELTA = {
    "[M+H]+": 1.007276, "[M+Na]+": 22.989218, "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823, "[M-H]-": -1.007276, "[M+Cl]-": 34.968853,
    "[M+CH2O2-H]-": 44.998201, "[M+C2H4O2-H]-": 59.013851,
    "[M]+": 0.0, "[M+2H]2+": 1.007276,  # approx, rare
    "[M-H2O+H]+": -17.003348, "[M-2H2O+H]+": -35.013913,
}
def neutral_mass(prec, adduct):
    # dimers: [2M+X]
    if adduct in ("[2M+H]+",): return (prec - 1.007276) / 2
    if adduct in ("[2M+Na]+",): return (prec - 22.989218) / 2
    if adduct in ("[2M-H]-",): return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    if d is None: return np.nan
    if adduct == "[M+2H]2+": return (prec - d) * 2  # rough
    return prec - d

def cosine(mz1, inten1, mz2, inten2, tol=0.02):
    # greedy matching within tol, intensities assumed normalized (base=1)
    if len(mz1) == 0 or len(mz2) == 0: return 0.0
    i = j = 0; num = 0.0
    a = np.asarray(inten1, dtype=float); b = np.asarray(inten2, dtype=float)
    # precompute norms
    na = float(np.sqrt((a*a).sum())); nb = float(np.sqrt((b*b).sum()))
    if na == 0 or nb == 0: return 0.0
    mz1 = np.asarray(mz1, dtype=float); mz2 = np.asarray(mz2, dtype=float)
    o1 = np.argsort(mz1); o2 = np.argsort(mz2)
    mz1, a = mz1[o1], a[o1]; mz2, b = mz2[o2], b[o2]
    while i < len(mz1) and j < len(mz2):
        d = mz1[i] - mz2[j]
        if abs(d) <= tol:
            num += a[i]*b[j]; i += 1; j += 1
        elif d < 0: i += 1
        else: j += 1
    return num / (na*nb)

print("load test...")
test = pd.read_parquet(TEST)
test["neutral"] = [neutral_mass(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
mol_neutral = test.groupby("molecule_id")["neutral"].median()
print(f"test mols {test['molecule_id'].nunique()} spectra {len(test)}")

print("load train (light)...")
train = pd.read_parquet(TRAIN, columns=["normalized_smiles","adduct","precursor_mz","ms2_mzs","ms2_normalized_intensities"])
train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
train = train[np.isfinite(train["neutral"].values)]
# structure table: median neutral mass per smiles
struct = train.groupby("normalized_smiles")["neutral"].median().reset_index()
struct["mass"] = struct["neutral"].values
print(f"unique structures {len(struct)}")

# For spectral scoring, keep up to 3 train spectra per structure (sample) to bound compute
print("sampling train spectra...")
train_sampled = train.groupby("normalized_smiles").head(3)[["normalized_smiles","ms2_mzs","ms2_normalized_intensities"]]

# index structures by mass for fast window query
struct = struct.sort_values("mass").reset_index(drop=True)
masses = struct["mass"].values
smiles_list = struct["normalized_smiles"].values
from bisect import bisect_left, bisect_right

def candidates(mass, ppm=20, min_n=200):
    tol = mass * ppm / 1e6
    lo = bisect_left(masses, mass - tol); hi = bisect_right(masses, mass + tol)
    # expand until min_n
    ppm_cur = ppm
    while hi - lo < min_n and ppm_cur < 500:
        ppm_cur *= 2; tol = mass * ppm_cur / 1e6
        lo = bisect_left(masses, mass - tol); hi = bisect_right(masses, mass + tol)
    return lo, hi, ppm_cur

# map smiles -> list of (mz,int)
spec_map = {}
for s, mz, inten in zip(train_sampled["normalized_smiles"], train_sampled["ms2_mzs"], train_sampled["ms2_normalized_intensities"]):
    spec_map.setdefault(s, []).append((np.asarray(mz, dtype=float), np.asarray(inten, dtype=float)))

rows = []
for mol, spectra in test.groupby("molecule_id"):
    qmass = float(mol_neutral.loc[mol])
    lo, hi, used_ppm = candidates(qmass)
    cands = smiles_list[lo:hi]
    # score each candidate by max cosine over molecule's spectra x candidate's sampled spectra
    qspecs = [(np.asarray(mz, dtype=float), np.asarray(it, dtype=float)) for mz, it in zip(spectra["ms2_mzs"], spectra["ms2_normalized_intensities"])]
    scored = []
    for s in cands[:2000]:  # cap for speed
        best = 0.0
        for qmz, qi in qspecs:
            for tmz, ti in spec_map.get(s, ()):
                c = cosine(qmz, qi, tmz, ti)
                if c > best: best = c
        # mass bonus: closer mass slightly better to break ties
        scored.append((best, s))
    scored.sort(reverse=True)
    top = [s for _, s in scored[:25]]
    # fallback fill with nearest-mass if short
    if len(top) < 25:
        extra = [s for s in cands if s not in set(top)][:25-len(top)]
        top += extra
    rows.append((mol, ";".join(top)))
    if len(rows) % 50 == 0: print(f"done {len(rows)}/400")

sub = pd.DataFrame(rows, columns=["molecule_id","smiles"])
sub.to_csv(OUT, index=False)
print(f"wrote {OUT} {sub.shape}")
