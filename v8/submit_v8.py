"""v8 production: cosine top-5 floor + fingerprint dot-product fills.
Writes submission.csv. Needs v8/fp_trans.pt weights (ships via fp dataset).
"""
import numpy as np, pandas as pd, os, pickle
from bisect import bisect_left, bisect_right
import torch
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine
from v2.train_fp import bin_spectrum, meta_vec, N_BINS, ADDUCTS, FpMLP

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


def window(masses, smi, qmass, ppm=20, min_n=200, cap=2000):
    tol = qmass * ppm / 1e6
    lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
    pcur = ppm
    while hi - lo < min_n and pcur < 500:
        pcur *= 2; tol = qmass * pcur / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
    return list(smi[lo:hi][:cap])


def window10(masses, smi, qmass, cap=3000):
    tol = max(qmass * 10 / 1e6, 0.01)
    lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
    return list(smi[lo:hi][:cap])


def main():
    device = "cpu"
    model = FpMLP(d_h=1536).to(device)
    model.load_state_dict(torch.load(f"{FP}/fp_trans.pt", map_location=device))
    model.eval()
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
    cf = pd.read_parquet(f"{FP}/coconut_fp.parquet")
    cfp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8)).astype(np.float32)
           for s, f in zip(cf["canonical_smiles"], cf["fp"])}
    co = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values
    tf = pd.read_parquet(f"{FP}/fingerprints.parquet")
    scol = "normalized_smiles" if "normalized_smiles" in tf.columns else "smiles"
    tfp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8)).astype(np.float32)
           for s, f in zip(tf[scol], tf["fp"])}

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
        # fingerprint channel over +-10ppm train+COCONUT
        feats = []
        for _, r in spectra.iterrows():
            v = np.zeros(N_BINS + len(ADDUCTS) + 2, dtype=np.float32)
            v[:N_BINS] = bin_spectrum(r["ms2_mzs"], r["ms2_normalized_intensities"])
            v[N_BINS:] = meta_vec(r["adduct"], r["precursor_mz"], r["collision_energy_ev"])
            feats.append(v)
        with torch.no_grad():
            Z = model(torch.from_numpy(np.stack(feats))).numpy()
        Zn = Z.mean(axis=0)
        Zn = Zn / (np.linalg.norm(Zn) + 1e-9)
        t10 = window10(tmass, tsmi, qmass)
        c10 = window10(cmass, csmi, qmass)
        scored = []
        for s in t10 + c10:
            f = tfp.get(s, cfp.get(s))
            if f is None: continue
            fn = f / (np.linalg.norm(f) + 1e-9)
            scored.append((float(fn @ Zn), s))
        scored.sort(reverse=True)
        seen = set(top5)
        out = list(top5) + [s for _, s in scored if not (s in seen or seen.add(s))][:20]
        rows.append((mol, ";".join(out[:25])))
        if (mi + 1) % 50 == 0: print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
