"""v4 production: analog propagation + entropy direct (train + COCONUT).
Writes submission.csv. Needs data/coconut_fp.parquet + fingerprints.parquet
via CASMI_FP (kernel: fingerprint dataset dir).
"""
import numpy as np, pandas as pd, os
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine
from v4.channels import entropy_similarity, tanimoto

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")
OUT = os.environ.get("CASMI_OUT", PROJECT)
FP = os.environ.get("CASMI_FP", f"{PROJECT}/data")
P_POW, DM_MAX = 3.0, 200.0


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


def main():
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
    tsamp = train.groupby("normalized_smiles").head(2).reset_index(drop=True)
    tneut = tsamp["neutral"].values

    cf = pd.read_parquet(f"{FP}/coconut_fp.parquet")
    cfp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8)) for s, f in zip(cf["canonical_smiles"], cf["fp"])}
    co = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values
    tf = pd.read_parquet(f"{FP}/fingerprints.parquet")
    scol = "normalized_smiles" if "normalized_smiles" in tf.columns else "smiles"
    tfp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8)) for s, f in zip(tf[scol], tf["fp"])}

    rows = []
    for mi, (mol, spectra) in enumerate(test.groupby("molecule_id")):
        qmass = float(mol_neutral.loc[mol])
        qspecs = [(np.asarray(mz, dtype=float), np.asarray(it, dtype=float))
                  for mz, it in zip(spectra["ms2_mzs"], spectra["ms2_normalized_intensities"])]
        tcands = window(tmass, tsmi, qmass)
        twin = tsamp[tsamp["normalized_smiles"].isin(set(tcands))]
        espec = {}
        for _, r in spectra.iterrows():
            for _, t in twin.iterrows():
                e = entropy_similarity(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                       t["ms2_mzs"], t["ms2_normalized_intensities"])
                k = t["normalized_smiles"]
                if e > espec.get(k, 0): espec[k] = e
        # analog pool: train spectra within DM_MAX Da
        dm = np.abs(tneut - qmass)
        pool = tsamp[dm <= DM_MAX]
        if len(pool) > 2000:
            pool = pool.sample(2000, random_state=mi)
        analogs = []
        for _, t in pool.iterrows():
            best = 0.0
            for qmz, qit in qspecs:
                e = entropy_similarity(qmz, qit, t["ms2_mzs"], t["ms2_normalized_intensities"])
                if e > best: best = e
            if best > 0.05: analogs.append((best, t["normalized_smiles"]))
        analogs.sort(reverse=True)
        analogs = analogs[:50]
        afps = [(s, tfp[s]) for _, s in analogs if s in tfp]
        ccands = window(cmass, csmi, qmass, cap=3000)
        scored = []
        for s in tcands + ccands:
            f = tfp.get(s, cfp.get(s))
            a_best = 0.0
            if f is not None:
                for sim, (_, g) in zip([a[0] for a in analogs], afps):
                    v = (sim ** P_POW) * tanimoto(f, g)
                    if v > a_best: a_best = v
            e = espec.get(s, 0.0)
            scored.append((max(a_best, 0.6 * e), s))
        scored.sort(reverse=True)
        seen, out = set(), []
        for _, s in scored:
            if s in seen: continue
            seen.add(s); out.append(s)
            if len(out) == 25: break
        rows.append((mol, ";".join(out)))
        if (mi + 1) % 50 == 0: print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
