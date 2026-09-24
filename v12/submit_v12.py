"""v12 production: v10 cleaned floor top-5 + formula-funnel/frag fills.
Writes submission.csv. Needs data/frag_cache.pkl. No RDKit at runtime.
"""
import numpy as np, pandas as pd, os, pickle
from bisect import bisect_left, bisect_right
from v10.submit_v10 import neutral_mass, denoise, window10
from v2.blend import cosine
from v_alt_score import fast_explained
from v7.submit_v7 import frag_match

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")
OUT = os.environ.get("CASMI_OUT", PROJECT)
FP = os.environ.get("CASMI_FP", f"{PROJECT}/data")


def main():
    with open(f"{FP}/frag_cache.pkl", "rb") as f:
        frags = pickle.load(f)
    test = pd.read_parquet(f"{IN}/test.parquet")
    test["neutral"] = [neutral_mass(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()
    train = pd.read_parquet(f"{IN}/train.parquet",
        columns=["normalized_smiles", "molecular_formula", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    tstruct = train.groupby("normalized_smiles").agg(
        mass=("neutral", "median"), formula=("molecular_formula", "first")).reset_index()
    tstruct = tstruct.sort_values("mass").reset_index(drop=True)
    tmass = tstruct["mass"].values
    tsmi = tstruct["normalized_smiles"].values
    tform = dict(zip(tstruct["normalized_smiles"], tstruct["formula"]))
    tsamp = train.groupby(["normalized_smiles", "adduct"]).head(2).reset_index(drop=True)
    cf = pd.read_parquet(f"{FP}/coconut_fp.parquet", columns=["canonical_smiles", "molecular_formula",
                                                              "exact_molecular_weight"])
    co = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values
    cform = dict(zip(co["canonical_smiles"], co["molecular_formula"]))

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
        # fills: formula funnel (top-2 formulae by explained) + frag fit
        ccands = window10(cmass, csmi, qmass)
        pool = list(dict.fromkeys(tcands[5:] + ccands))
        fscore = {}
        for s in pool:
            f = tform.get(s, cform.get(s, ""))
            if not f:
                continue
            best = 0.0
            for _, r in spectra.iterrows():
                e, _ = fast_explained(r["ms2_mzs"], r["ms2_normalized_intensities"], f, r["adduct"])
                if e > best:
                    best = e
            fscore[s] = best
        # top-2 formulae
        fbest = {}
        for s, v in fscore.items():
            f = tform.get(s, cform.get(s, ""))
            if v > fbest.get(f, (0, ""))[0]:
                pass
        forder = sorted(set(tform.get(s, cform.get(s, "")) for s in pool if tform.get(s, cform.get(s, ""))),
                        key=lambda f: max([fscore.get(s, 0) for s in pool
                                           if tform.get(s, cform.get(s, "")) == f] or [0]),
                        reverse=True)[:2]
        fills = []
        for s in pool:
            if tform.get(s, cform.get(s, "")) not in forder:
                continue
            fr = frags.get(s, [])
            best = 0.0
            for _, r in spectra.iterrows():
                v = frag_match(r["ms2_mzs"], r["ms2_normalized_intensities"], fr, r["adduct"])
                if v > best:
                    best = v
            fills.append((best, s))
        fills.sort(reverse=True)
        seen = set(top5)
        out = list(top5) + [s for _, s in fills if not (s in seen or seen.add(s))][:20]
        rows.append((mol, ";".join(out[:25])))
        if (mi + 1) % 50 == 0:
            print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
