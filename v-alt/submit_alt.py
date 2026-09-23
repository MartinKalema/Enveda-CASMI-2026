"""v-alt production: formula-first isomer retrieval, NO cosine floor anywhere.

Per test molecule: mass window -> rank formulae by subformula explained-intensity
-> rank top-formula isomers by forward fragmentation fit (+ loss-bonus tiebreak).
Writes submission.csv. Needs v-alt/frag_disk.pkl (grows on first run; RDKit required).
Env: CASMI_IN / CASMI_OUT / CASMI_FP like other versions.
"""
import numpy as np
import pandas as pd
import os
import importlib.util
from bisect import bisect_left, bisect_right

_SPEC = importlib.util.spec_from_file_location(
    "alt_score", os.path.join(os.path.dirname(os.path.abspath(__file__)), "alt_score.py"))
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)
(neutral_mass, fast_explained, loss_bonus, frag_fit,
 load_frag_disk, save_frag_disk) = (
    _mod.neutral_mass, _mod.fast_explained, _mod.loss_bonus,
    _mod.frag_fit, _mod.load_frag_disk, _mod.save_frag_disk)

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")
OUT = os.environ.get("CASMI_OUT", PROJECT)
FP = os.environ.get("CASMI_FP", f"{PROJECT}/data")
TOPK_FORM = int(os.environ.get("ALT_TOPK_FORM", "3"))
POOL_CAP = int(os.environ.get("ALT_POOL_CAP", "200"))
WITH_COCONUT = os.environ.get("ALT_COCONUT", "0") == "1"


def window(masses, smi, qmass, ppm=20, min_n=100, cap=400):
    tol = qmass * ppm / 1e6
    lo = bisect_left(masses, qmass - tol)
    hi = bisect_right(masses, qmass + tol)
    pcur = ppm
    while hi - lo < min_n and pcur < 500:
        pcur *= 2
        tol = qmass * pcur / 1e6
        lo = bisect_left(masses, qmass - tol)
        hi = bisect_right(masses, qmass + tol)
    return list(smi[lo:hi][:cap])


def main():
    load_frag_disk()
    test = pd.read_parquet(f"{IN}/test.parquet")
    test["neutral"] = [neutral_mass(p, a) for p, a in
                       zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()
    train = pd.read_parquet(
        f"{IN}/train.parquet",
        columns=["normalized_smiles", "molecular_formula", "precursor_mz",
                 "adduct", "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in
                        zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    tstruct = train.groupby("normalized_smiles").agg(
        mass=("neutral", "median"), formula=("molecular_formula", "first")).reset_index()
    tstruct = tstruct.sort_values("mass").reset_index(drop=True)
    tmass = tstruct["mass"].values
    tsmi = tstruct["normalized_smiles"].values
    tform = dict(zip(tstruct["normalized_smiles"], tstruct["formula"]))
    cmass = csmi = cf = None
    if WITH_COCONUT:
        co = pd.read_parquet(f"{FP}/coconut_fp.parquet",
                             columns=["canonical_smiles", "exact_molecular_weight",
                                      "molecular_formula"])
        co = co.sort_values("exact_molecular_weight").reset_index(drop=True)
        cmass = co["exact_molecular_weight"].values
        csmi = co["canonical_smiles"].values
        cf = dict(zip(co["canonical_smiles"], co["molecular_formula"]))

    rows = []
    for mi, (mol, spectra) in enumerate(test.groupby("molecule_id")):
        qmass = float(mol_neutral.loc[mol])
        qrows = list(spectra.head(3).iterrows())
        cands = window(tmass, tsmi, qmass)
        fmap = tform
        if WITH_COCONUT:
            cc = window(cmass, csmi, qmass)
            cands = cands + [s for s in cc if s not in set(cands)]
            fmap = dict(tform)
            fmap.update(cf)
        forms = sorted(set(fmap[s] for s in cands if s in fmap))
        fscore = {}
        for f in forms:
            best = 0.0
            for _, r in qrows:
                v = fast_explained(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                   f, r["adduct"])
                if v > best:
                    best = v
            fscore[f] = best
        topf = set(sorted(fscore, key=fscore.get, reverse=True)[:TOPK_FORM])
        mmap = dict(zip(tstruct["normalized_smiles"], tstruct["mass"]))
        if WITH_COCONUT:
            mmap.update(dict(zip(co["canonical_smiles"], co["exact_molecular_weight"])))
        pool = [s for s in cands if fmap.get(s) in topf]
        pool = sorted(pool, key=lambda s: abs(float(mmap.get(s, qmass)) - qmass))[:POOL_CAP]
        scored = []
        for s in pool:
            f = fmap.get(s, "")
            bf = bg = bl = 0.0
            for _, r in qrows:
                v = frag_fit(r["ms2_mzs"], r["ms2_normalized_intensities"], s, r["adduct"])
                if v > bf:
                    bf = v
                v = fast_explained(r["ms2_mzs"], r["ms2_normalized_intensities"], f, r["adduct"])
                if v > bg:
                    bg = v
                v = loss_bonus(r["ms2_mzs"], r["ms2_normalized_intensities"], qmass, r["adduct"])
                if v > bl:
                    bl = v
            scored.append((0.5 * bf + 0.3 * bg + 0.2 * bl, s))
        scored.sort(reverse=True)
        # fill leftovers (funneled-out, formula order) so slots are never empty
        rest = [s for s in cands if s not in set(x[1] for x in scored)]
        rest.sort(key=lambda s: fscore.get(fmap.get(s, ""), 0.0), reverse=True)
        out = [s for _, s in scored] + rest
        rows.append((mol, ";".join(out[:25])))
        if (mi + 1) % 25 == 0:
            save_frag_disk()
            print(f"done {mi+1}/400", flush=True)
    save_frag_disk()
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(
        f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
