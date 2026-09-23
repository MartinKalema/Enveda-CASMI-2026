"""v9 production: cosine top-5 floor + GBM-25 fills (mass/formula/tanimoto).
Writes submission.csv. Needs v8/gbm25.pkl (ships via fp dataset).
"""
import numpy as np, pandas as pd, os, pickle
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine
from v4.channels import entropy_similarity, tanimoto

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
    with open(f"{FP}/gbm25.pkl", "rb") as f:
        _m = pickle.load(f)
    ranker, FEATS = _m["model"], _m["feats"]
    test = pd.read_parquet(f"{IN}/test.parquet")
    test["neutral"] = [neutral_mass(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()
    train = pd.read_parquet(f"{IN}/train.parquet",
        columns=["normalized_smiles", "molecular_formula", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
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
    _smas = dict(zip(tstruct.index, tstruct.values))
    _smas.update(dict(zip(co["canonical_smiles"], co["exact_molecular_weight"])))
    _trf = train.groupby("normalized_smiles")["molecular_formula"].first()
    _sform = dict(_trf)
    _sform.update(dict(zip(co["canonical_smiles"], co["molecular_formula"])))
    from collections import Counter as _Counter
    _fprior = _Counter(train["molecular_formula"].tolist())

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
        _cspec = dict((s, v) for v, s in tscored)
        # GBM-25 fills: cos/ent/ana + mass_err + log_prior + t_top1
        espec = {}
        for _, r in spectra.iterrows():
            for _, t in twin.iterrows():
                e = entropy_similarity(r["ms2_mzs"], r["ms2_normalized_intensities"],
                        t["ms2_mzs"], t["ms2_normalized_intensities"])
                k = t["normalized_smiles"]
                if e > espec.get(k, 0):
                    espec[k] = e
        tneu = tsamp["neutral"].values
        dm = np.abs(tneu - qmass)
        _pool = tsamp[dm <= 200.0]
        if len(_pool) > 2000:
            _pool = _pool.sample(2000, random_state=mi)
        _an, _af = [], []
        for _, t in _pool.iterrows():
            _b = 0.0
            for qmz, qit in qspecs:
                _e = entropy_similarity(qmz, qit, t["ms2_mzs"], t["ms2_normalized_intensities"])
                if _e > _b:
                    _b = _e
            if _b > 0.05 and t["normalized_smiles"] in tfp:
                _an.append(_b)
                _af.append(tfp[t["normalized_smiles"]])
        _ord = np.argsort(-np.array(_an))[:15] if _an else []
        _top = [(float(_an[i]), _af[i]) for i in _ord]
        t10 = window10(tmass, tsmi, qmass)
        c10 = window10(cmass, csmi, qmass)
        _rows, _order = [], []
        for s in t10 + c10:
            f = tfp.get(s, cfp.get(s))
            _ab = 0.0
            if f is not None:
                for _sim, _g in _top:
                    _v = (_sim ** 3.0) * tanimoto(f > 0.5, _g > 0.5)
                    if _v > _ab:
                        _ab = _v
            _cm = _smas.get(s, np.nan)
            _rows.append([_cspec.get(s, 0.0), espec.get(s, 0.0), _ab,
                          abs(_cm - qmass) / qmass if _cm == _cm else 1.0,
                          float(np.log1p(_fprior.get(_sform.get(s, ""), 0))),
                          max([tanimoto(f > 0.5, _g > 0.5) for _, _g in _top] or [0.0])
                          if f is not None else 0.0])
            _order.append(s)
        _proba = ranker.predict_proba(np.array(_rows, dtype=np.float32))[:, 1]
        scored = sorted(zip(_proba, _order), reverse=True)
        seen = set(top5)
        out = list(top5) + [s for _, s in scored if not (s in seen or seen.add(s))][:20]
        rows.append((mol, ";".join(out[:25])))
        if (mi + 1) % 50 == 0: print(f"done {mi+1}/400", flush=True)
    pd.DataFrame(rows, columns=["molecule_id", "smiles"]).to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
