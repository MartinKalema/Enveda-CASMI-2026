"""Precompute MetFrag fragment masses for the candidate universe (multiprocess).
Union of train+COCONUT mass windows over all test molecules -> data/frag_cache.pkl
"""
import numpy as np, pandas as pd, os, pickle
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
from v1.subformula import ADDUCT_DELTA
from v7.metfrag import fragment_masses

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def window(masses, smi, qmass, ppm=20, min_n=200, cap=2000):
    tol = qmass * ppm / 1e6
    lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
    pcur = ppm
    while hi - lo < min_n and pcur < 500:
        pcur *= 2; tol = qmass * pcur / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
    return list(smi[lo:hi][:cap])


def _frag_one(s):
    try:
        return s, sorted(fragment_masses(s))
    except Exception:
        return s, []


def main(nw=10):
    test = pd.read_parquet(f"{IN}/test.parquet")
    test["neutral"] = [neutral_mass(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()
    train = pd.read_parquet(f"{IN}/train.parquet", columns=["normalized_smiles", "precursor_mz", "adduct"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    tstruct = train.groupby("normalized_smiles")["neutral"].median()
    tmass = tstruct.sort_values().values
    tsmi = tstruct.sort_values().index.values
    cf = pd.read_parquet(f"{IN}/coconut_fp.parquet", columns=["canonical_smiles", "exact_molecular_weight"])
    co = cf.sort_values("exact_molecular_weight")
    cmass, csmi = co["exact_molecular_weight"].values, co["canonical_smiles"].values
    uni = set()
    for mol in mol_neutral.index:
        qmass = float(mol_neutral.loc[mol])
        uni.update(window(tmass, tsmi, qmass))
        uni.update(window(cmass, csmi, qmass, cap=3000))
    uni = sorted(uni)
    print(f"universe: {len(uni)}", flush=True)
    out = {}
    done = 0
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for s, frags in ex.map(_frag_one, uni, chunksize=50):
            out[s] = frags
            done += 1
            if done % 20000 == 0: print(f"frag {done}/{len(uni)}", flush=True)
    with open(f"{PROJECT}/data/frag_cache.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"saved {len(out)} fragment sets", flush=True)


if __name__ == "__main__":
    main()
