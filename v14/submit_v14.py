"""v14 production: v10 floor top-5 + THEIR full stack fills.
Writes submission.csv. Needs (fp dataset): fp_single_s2.pt, their_ranker.pkl,
their_fp_prod.pkl, their_frag_prod.pkl, coconut_fp.parquet, fingerprints.parquet.
"""
import numpy as np
import pandas as pd
import os
import pickle
import torch
from bisect import bisect_left, bisect_right
import gc
from v13.fork_frag_raw import explain_score as their_explain

from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")
OUT = os.environ.get("CASMI_OUT", PROJECT)
FP = os.environ.get("CASMI_FP", f"{PROJECT}/data")
TOP_N = 200
INT_FLOOR = 0.01


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+":
        return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+":
        return (prec - 22.989218) / 2
    if adduct == "[2M-H]-":
        return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def denoise(mz, it, n=TOP_N, floor=INT_FLOOR):
    mz = np.asarray(mz, dtype=float)
    it = np.asarray(it, dtype=float)
    keep = it >= floor
    mz, it = mz[keep], it[keep]
    if len(mz) > n:
        o = np.argsort(-it)[:n]
        mz, it = mz[o], it[o]
    o = np.argsort(mz)
    return mz[o], it[o]


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
    import v14.libsearch as L
    from v14.loaders import load_library_df
    from v14.rank import rank_features
    import v13.fork_fpnet_raw as F
    device = "cpu"
    ck = torch.load(f"{FP}/fp_single_s2.pt", map_location="cpu", weights_only=False)
    tnet = F.FPNet(ck["nbits"], d=ck["d"], layers=ck["layers"]).eval()
    tnet.load_state_dict(ck["model"])
    F._MODEL = ([tnet], [], "cpu", ck["nbits"])
    rankers = pickle.load(open(f"{PROJECT}/v13/their_ranker.pkl", "rb"))
    with open(f"{FP}/their_fp_keys.pkl", "rb") as f:
        _keys = pickle.load(f)
    _M = np.load(f"{FP}/their_fp_packed.npy", mmap_mode="r")
    _k2i = {s: i for i, s in enumerate(_keys)}

    def _bits(s):
        i = _k2i.get(s)
        if i is None:
            return None
        return np.unpackbits(_M[i]).astype(np.float32)[:6930]
    with open(f"{FP}/their_frag_prod.pkl", "rb") as f:
        uni_fr = pickle.load(f)
    test = pd.read_parquet(f"{IN}/test.parquet")
    test["neutral"] = [neutral_mass(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()
    train = pd.read_parquet(
        f"{IN}/train.parquet",
        columns=["normalized_smiles", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    tstruct = train.groupby("normalized_smiles")["neutral"].median()
    tmass = tstruct.sort_values().values
    tsmi = tstruct.sort_values().index.values
    tsamp = train.groupby(["normalized_smiles", "adduct"]).head(2).reset_index(drop=True)
    cf = pd.read_parquet(f"{FP}/coconut_fp.parquet", columns=["canonical_smiles", "exact_molecular_weight"])
    co = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    del cf
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values
    rows = []
    for mi, (mol, spectra) in enumerate(test.groupby("molecule_id")):
        qmass = float(mol_neutral.loc[mol])
        tcands = window10(tmass, tsmi, qmass)
        twin = tsamp[tsamp["normalized_smiles"].isin(set(tcands))]
        qspecs = {}
        for _, r in spectra.iterrows():
            qspecs.setdefault(r["adduct"], []).append(
                denoise(r["ms2_mzs"], r["ms2_normalized_intensities"]))
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
        # fills: their 31-feature stack
        ccands = window10(cmass, csmi, qmass)
        pool = list(dict.fromkeys(tcands[5:] + ccands))
        lib = load_library_df(
            train[train["normalized_smiles"].isin(set(pool))], L.neutral_mass)
        rep, rep_key, rep_nm, rep_ad = L.build_rep(lib)
        q0 = spectra.iloc[0]
        specs = [(q0["ms2_mzs"], q0["ms2_normalized_intensities"], q0["adduct"])]
        lib_hits = L.lib_sim(lib, specs, qmass)
        lv = np.array([lib_hits.get(s, 0.0) for s in pool], np.float32)
        an = L.analog_sim(lib, specs, qmass, rep, rep_key, rep_nm, rep_ad)
        afp, asim = [], []
        for k, v in an[:80]:
            t = _bits(k)
            if t is not None:
                afp.append(t)
                asim.append(v)
        afp = np.stack(afp) if afp else None
        asim = np.array(asim, np.float32)
        ce = q0["collision_energy_ev"]
        try:
            ce = float(np.mean(np.atleast_1d(ce))) if ce is not None and len(np.atleast_1d(ce)) else 25.0
        except Exception:
            ce = 25.0
        zlog = F._logits_raw([(q0["ms2_mzs"], q0["ms2_normalized_intensities"])], [tnet],
                             q0["precursor_mz"], q0["adduct"], "timsTOF", ce, 1.0)
        cfp = []
        for s in pool:
            t = _bits(s)
            cfp.append(t if t is not None else np.zeros(6930, np.float32))
        cfp = np.stack(cfp)
        fr = []
        for s in pool:
            f = uni_fr.get(s, np.zeros(0))
            m2 = np.asarray(q0["ms2_mzs"], float)
            i2 = np.asarray(q0["ms2_normalized_intensities"], float)
            md = 1.0 if str(q0["adduct"]).rstrip().endswith("]+") else -1.0
            fr.append(float(their_explain(np.asarray(f, float), m2, i2,
                                            mode=md, tol=0.01)) if len(f) else 0.0)
        fr = np.array(fr, np.float32)
        X = rank_features(cfp, lv, afp, asim, model_logits=zlog, frag=fr)
        pt = np.mean([m.predict_proba(X)[:, 1] for m in rankers.values()], axis=0)
        order = sorted(zip(pt, pool), reverse=True)
        seen = set(top5)
        out = list(top5) + [s for _, s in order if not (s in seen or seen.add(s))][:20]
        rows.append((mol, ";".join(out[:25])))
        del lib
        gc.collect()
        if (mi + 1) % 50 == 0:
            print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
