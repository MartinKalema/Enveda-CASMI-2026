"""Build candidate pool from ORIGINAL COCONUT Oct-2026 dump.
No quality filtering (isolates pool-version effect); dedupe + parse check only.
In:  /tmp/coconut-orig/coconut_csv_lite-10-2026.csv
Out: data/coconut_orig.parquet (canonical_smiles, exact_molecular_weight,
     annotation_level), data/orig_universe.pkl (sorted SMILES over test windows)
"""
import pickle

import numpy as np
import pandas as pd
from bisect import bisect_left, bisect_right

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
ORIG = "/tmp/coconut-orig/coconut_csv_lite-10-2026.csv"


def window10(masses, smi, qmass, cap=3000, min_n=50):
    tol = max(qmass * 10 / 1e6, 0.01)
    lo = bisect_left(masses, qmass - tol)
    hi = bisect_right(masses, qmass + tol)
    mult = 10.0
    while hi - lo < min_n and mult < 5000:
        mult *= 2
        tol = max(qmass * 10 / 1e6, 0.01) * mult / 10.0
        lo = bisect_left(masses, qmass - tol)
        hi = bisect_right(masses, qmass + tol)
    return list(smi[lo:hi][:cap])


def main():
    from v14.precompute_prod import nm
    usecols = ["canonical_smiles", "exact_molecular_weight", "annotation_level",
               "molecular_formula"]
    parts = []
    for ch in pd.read_csv(ORIG, usecols=usecols, chunksize=200000):
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    print("raw rows:", len(df), flush=True)
    df = df[df["canonical_smiles"].notna() & (df["canonical_smiles"] != "")]
    df = df[np.isfinite(df["exact_molecular_weight"].values)]
    df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)
    print("deduped:", len(df), "ann:", df["annotation_level"].value_counts().to_dict(),
          flush=True)
    df[["canonical_smiles", "exact_molecular_weight", "annotation_level"]].to_parquet(
        f"{PROJECT}/data/coconut_orig.parquet", index=False)
    # RDKit parse + mass-agreement spot check
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    samp = df.sample(min(2000, len(df)), random_state=0)
    ok = 0
    errs = []
    for s, m in zip(samp["canonical_smiles"], samp["exact_molecular_weight"]):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            ok += 1
            errs.append(abs(ExactMolWt(mol) - m) / max(m, 1e-9) * 1e6)
        except Exception:
            pass
    print(f"parse rate: {ok}/{len(samp)}, mass median abs err ppm: "
          f"{float(np.median(errs)):.2f}", flush=True)
    # production universe over test molecules (same windows as submit)
    test = pd.read_parquet(f"{PROJECT}/data/test.parquet")
    test["neutral"] = [nm(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()
    train = pd.read_parquet(f"{PROJECT}/data/train.parquet",
                            columns=["normalized_smiles", "precursor_mz", "adduct"])
    train["neutral"] = [nm(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    tstruct = train.groupby("normalized_smiles")["neutral"].median()
    tmass = tstruct.sort_values().values
    tsmi = tstruct.sort_values().index.values
    co = df.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values
    uni = set()
    for mol in mol_neutral.index:
        qmass = float(mol_neutral.loc[mol])
        uni.update(window10(tmass, tsmi, qmass))
        uni.update(window10(cmass, csmi, qmass))
    uni = sorted(uni)
    pickle.dump(uni, open(f"{PROJECT}/data/orig_universe.pkl", "wb"))
    print("universe:", len(uni), flush=True)


if __name__ == "__main__":
    main()
