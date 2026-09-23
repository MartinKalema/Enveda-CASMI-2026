"""v10 production: cleaned cosine floor (per-adduct index, denoised peaks,
+-10ppm hybrid windows) + v8 fingerprint fills. Writes submission.csv.
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
TOP_N = 200
INT_FLOOR = 0.01


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def denoise(mz, it, n=TOP_N, floor=INT_FLOOR):
    mz = np.asarray(mz, dtype=float); it = np.asarray(it, dtype=float)
    keep = it >= floor
    mz, it = mz[keep], it[keep]
    if len(mz) > n:
        o = np.argsort(-it)[:n]
        mz, it = mz[o], it[o]
    o = np.argsort(mz)
    return mz[o], it[o]


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
    # per-(structure, adduct) index: same-adduct cosine only (0.445 vs 0.058)
    tstruct = train.groupby("normalized_smiles")["neutral"].median()
    tmass = tstruct.sort_values().values
    tsmi = tstruct.sort_values().index.values
    tsamp = train.groupby(["normalized_smiles", "adduct"]).head(2).reset_index(drop=True)
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
        tcands = window10(tmass, tsmi, qmass)
        twin = tsamp[tsamp["normalized_smiles"].isin(set(tcands))]
        qspecs = {}
        for _, r in spectra.iterrows():
            qspecs.setdefault(r["adduct"], []).append(denoise(r["ms2_mzs"], r["ms2_normalized_intensities"]))
        tscored = []
        for s in tcands:
            best = 0.0
            sub = twin[twin["normalized_smiles"] == s]
            for ad, ql in qspecs.items():
                suba = sub[sub["adduct"] == ad]
                if len(suba) == 0:
                    suba = sub
                for tmz, tit in zip(suba["ms2_mzs"], suba["ms2_normalized_intensities"]):
                    dmz, dit = denoise(tmz, tit)
                    for qmz, qit in ql:
                        c = cosine(qmz, qit, dmz, dit)
                        if c > best:
                            best = c
            tscored.append((best, s))
        tscored.sort(reverse=True)
        top5 = [s for _, s in tscored[:5]]
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
        c10 = window10(cmass, csmi, qmass)
        scored = []
        for s in tcands[:1500] + c10:
            f = tfp.get(s, cfp.get(s))
            if f is None:
                continue
            fn = f / (np.linalg.norm(f) + 1e-9)
            scored.append((float(fn @ Zn), s))
        scored.sort(reverse=True)
        seen = set(top5)
        out = list(top5) + [s for _, s in scored if not (s in seen or seen.add(s))][:20]
        rows.append((mol, ";".join(out[:25])))
        if (mi + 1) % 50 == 0:
            print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
