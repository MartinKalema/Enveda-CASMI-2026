"""v6 production: cosine top-5 floor + GBM-ranked fills.
Writes submission.csv. Needs v5/gbm_ranker.pkl in FP dir + sklearn.
"""
import numpy as np, pandas as pd, os, pickle
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine, tanimoto
from v4.channels import entropy_similarity

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")
OUT = os.environ.get("CASMI_OUT", PROJECT)
FP = os.environ.get("CASMI_FP", f"{PROJECT}/data")
TOP_N = 150
P_POW, DM_MAX = 3.0, 200.0
FEATS = ["cos", "ent", "ana"]


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


def main():
    with open(f"{FP}/gbm_ranker.pkl", "rb") as f:
        ranker = pickle.load(f)["model"]
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
        tcands = window(tmass, tsmi, qmass)
        twin = tsamp[tsamp["normalized_smiles"].isin(set(tcands))]
        qspecs = [topn(np.asarray(mz, dtype=float), np.asarray(it, dtype=float))
                  for mz, it in zip(spectra["ms2_mzs"], spectra["ms2_normalized_intensities"])]
        cspec, espec = {}, {}
        for _, r in spectra.iterrows():
            for _, t in twin.iterrows():
                e = entropy_similarity(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                       t["ms2_mzs"], t["ms2_normalized_intensities"])
                c = cosine(*topn(np.asarray(r["ms2_mzs"], dtype=float),
                                 np.asarray(r["ms2_normalized_intensities"], dtype=float)),
                           *topn(np.asarray(t["ms2_mzs"], dtype=float),
                                 np.asarray(t["ms2_normalized_intensities"], dtype=float)))
                k = t["normalized_smiles"]
                if e > espec.get(k, 0): espec[k] = e
                if c > cspec.get(k, 0): cspec[k] = c
        cos_top5 = [s for _, s in sorted(((cspec.get(s, 0.0), s) for s in tcands), reverse=True)[:5]]
        dm = np.abs(tneut - qmass)
        pool = tsamp[dm <= DM_MAX]
        if len(pool) > 2000: pool = pool.sample(2000, random_state=mi)
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
        asims = [a[0] for a in analogs[:len(afps)]]
        ccands = window(cmass, csmi, qmass, cap=3000)
        feats, order = [], []
        for s in tcands + ccands:
            f = tfp.get(s, cfp.get(s))
            ab = 0.0
            if f is not None:
                for sim, (_, g) in zip(asims, afps):
                    v = (sim ** P_POW) * tanimoto(f, g)
                    if v > ab: ab = v
            feats.append([cspec.get(s, 0.0), espec.get(s, 0.0), ab])
            order.append(s)
        proba = ranker.predict_proba(np.array(feats, dtype=np.float32))[:, 1]
        ranked = sorted(zip(proba, order), reverse=True)
        seen = set(cos_top5)
        out = list(cos_top5)
        for _, s in ranked:
            if s in seen: continue
            seen.add(s); out.append(s)
            if len(out) == 25: break
        rows.append((mol, ";".join(out[:25])))
        if (mi + 1) % 50 == 0: print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
