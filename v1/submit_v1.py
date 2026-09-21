"""v1 production: explained-intensity ranking + formula prior + fusion.
Writes submission.csv (400 x 25). RDKit-free, CPU.
"""
import numpy as np, pandas as pd, os
from bisect import bisect_left, bisect_right
from .subformula import explained_intensity, ADDUCT_DELTA, subformula_masses
from .formula import FormulaPrior

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")
OUT = os.environ.get("CASMI_OUT", PROJECT)


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def main():
    test = pd.read_parquet(f"{IN}/test.parquet")
    test["neutral"] = [neutral_mass(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()

    train = pd.read_parquet(f"{IN}/train.parquet",
        columns=["normalized_smiles", "molecular_formula", "adduct",
                 "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    prior = FormulaPrior()
    prior.fit(train["molecular_formula"].tolist())

    struct = train.groupby("normalized_smiles").agg(
        mass=("neutral", "median"), formula=("molecular_formula", "first")).reset_index()
    struct = struct.sort_values("mass").reset_index(drop=True)
    masses = struct["mass"].values
    smi = struct["normalized_smiles"].values
    fmap = dict(zip(struct["normalized_smiles"], struct["formula"]))
    # pre-warm subformula cache for common formulae in test mass range (245-460 Da)
    priors = {f: prior.score(f) for f in struct["formula"].unique()}

    rows = []
    for mi, (mol, spectra) in enumerate(test.groupby("molecule_id")):
        qmass = float(mol_neutral.loc[mol])
        tol = qmass * 20 / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        pcur = 20
        while hi - lo < 200 and pcur < 500:
            pcur *= 2; tol = qmass * pcur / 1e6
            lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        cands = smi[lo:hi][:2000]
        qspecs = list(zip(spectra["ms2_mzs"], spectra["ms2_normalized_intensities"], spectra["adduct"]))
        scored = []
        for s in cands:
            f = fmap[s]
            subformula_masses(f)  # warm cache
            best = 0.0
            for mz, it, ad in qspecs:
                v = explained_intensity(mz, it, f, ad)[0]
                if v > best: best = v
            scored.append((best + 0.02 * priors.get(f, -10.0), best, s))
        scored.sort(reverse=True)
        top = [s for _, _, s in scored[:25]]
        rows.append((mol, ";".join(top)))
        if (mi + 1) % 50 == 0: print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
