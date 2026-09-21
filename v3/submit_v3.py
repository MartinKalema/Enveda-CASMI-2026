"""v3: train-cosine top-5 (floor) + COCONUT NP fills (recall lottery).
Peak renormalization (top-150) per limits analysis. RDKit-free at runtime
(fingerprints precomputed). Writes submission.csv.
"""
import numpy as np, pandas as pd, os
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine, tanimoto

_fp_cache = {}


def train_fp_lookup():
    if "tfp" not in _fp_cache:
        df = pd.read_parquet(f"{FP}/fingerprints.parquet")
        _fp_cache["tfp"] = {s: np.unpackbits(np.asarray(f, dtype=np.uint8))
                            for s, f in zip(df["normalized_smiles"] if "normalized_smiles" in df.columns else df["smiles"], df["fp"])}
    return _fp_cache["tfp"]

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")
OUT = os.environ.get("CASMI_OUT", PROJECT)
FP = os.environ.get("CASMI_FP", f"{PROJECT}/data")
TOP_N = 150


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
    tsamp = train.groupby("normalized_smiles").head(2)

    coco = pd.read_parquet(f"{FP}/coconut_fp.parquet")
    cmass = coco["exact_molecular_weight"].values
    co = coco.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = co["exact_molecular_weight"].values
    cfp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8))
           for s, f in zip(co["canonical_smiles"], co["fp"])}
    cform = dict(zip(co["canonical_smiles"], co["molecular_formula"]))
    cmass_d = dict(zip(co["canonical_smiles"], co["exact_molecular_weight"]))
    from collections import Counter
    form_prior = Counter(co["molecular_formula"].tolist())
    tfp = train_fp_lookup()  # train smiles -> fp bits

    rows = []
    for mi, (mol, spectra) in enumerate(test.groupby("molecule_id")):
        qmass = float(mol_neutral.loc[mol])
        tol = qmass * 20 / 1e6
        lo = bisect_left(tmass, qmass - tol); hi = bisect_right(tmass, qmass + tol)
        pcur = 20
        while hi - lo < 200 and pcur < 500:
            pcur *= 2; tol = qmass * pcur / 1e6
            lo = bisect_left(tmass, qmass - tol); hi = bisect_right(tmass, qmass + tol)
        tcands = list(tsmi[lo:hi][:2000])
        twin = tsamp[tsamp["normalized_smiles"].isin(set(tcands))]
        qspecs = [(np.asarray(mz, dtype=float), np.asarray(it, dtype=float))
                  for mz, it in zip(spectra["ms2_mzs"], spectra["ms2_normalized_intensities"])]
        qspecs = [topn(mz, it) for mz, it in qspecs]
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
        # mean fp of train top-5 as NP-analog query
        vecs = [tfp[s] for s in top5 if s in tfp]
        qfp = (np.stack(vecs).mean(axis=0) > 0.5) if vecs else None
        # COCONUT mass window
        ctol = qmass * 20 / 1e6
        clo = bisect_left(cmass, qmass - ctol); chi = bisect_right(cmass, qmass + ctol)
        cpcur = 20
        while chi - clo < 200 and cpcur < 500:
            cpcur *= 2; ctol = qmass * cpcur / 1e6
            clo = bisect_left(cmass, qmass - ctol); chi = bisect_right(cmass, qmass + ctol)
        ccands = co.iloc[clo:chi]["canonical_smiles"].tolist()[:3000]
        seen = set(top5)
        cscored = []
        for s in ccands:
            if s in seen: continue
            f = cfp.get(s)
            t = tanimoto(qfp, f) if (qfp is not None and f is not None) else 0.0
            prior = np.log1p(form_prior.get(cform.get(s, ""), 0))
            merr = abs(cmass_d[s] - qmass) / qmass
            cscored.append((t + 0.05 * prior - merr, s))
        cscored.sort(reverse=True)
        fills = [s for _, s in cscored[:20] if not (s in seen or seen.add(s))]
        # backbone: remaining train-cosine ranks fill any leftover
        rest = [s for _, s in tscored[5:] if s not in seen]
        out = (top5 + fills + rest)[:25]
        rows.append((mol, ";".join(out)))
        if (mi + 1) % 50 == 0: print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
