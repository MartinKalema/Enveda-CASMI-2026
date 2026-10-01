"""Single-variable bout: THEIR 31-feature stack + ranker, fr masses differ.
Arm A: their unweighted frag masses. Arm B: OUR BDE frag masses.
Scorer (explain_score), fingerprint, ranker, queries identical.
Usage: .venv/bin/python -m v14.fr_bout [n_query]
Writes v14/fr_bout_log.csv with per-seed, per-arm reciprocal ranks.
"""
import gc
import pickle
import sys

import numpy as np
import pandas as pd
import torch

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)


def nm(p, a):
    from v1.subformula import ADDUCT_DELTA
    if a == "[2M+H]+":
        return (p - 1.007276) / 2
    if a == "[2M+Na]+":
        return (p - 22.989218) / 2
    if a == "[2M-H]-":
        return (p + 1.007276) / 2
    d = ADDUCT_DELTA.get(a)
    return p - d if d is not None else np.nan


def main(n_query=None):
    import v14.libsearch as L
    from v14.loaders import load_library_df
    from v14.rank import rank_features
    import v13.fork_fpnet_raw as F
    from v13.fork_frag_raw import explain_score as their_explain
    from v13.frag_up import cached_fragments_bde
    from v13.frag_up import _regime as bde_regime

    ck = torch.load("/tmp/fpmodels/fp_single_s2.pt", map_location="cpu", weights_only=False)
    tnet = F.FPNet(ck["nbits"], d=ck["d"], layers=ck["layers"]).eval()
    tnet.load_state_dict(ck["model"])
    F._MODEL = ([tnet], [], "cpu", ck["nbits"])
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    F.MACCSkeys = MACCSkeys
    F.ExactMolWt = ExactMolWt
    F.BITS = np.load("/tmp/cocofp/fp_bits.npy")
    F._fp_init()
    torch.set_num_threads(4)

    rankers = pickle.load(open(f"{PROJECT}/v13/their_ranker.pkl", "rb"))
    rankers_bde = pickle.load(open(f"{PROJECT}/v14/ranker_bde.pkl", "rb"))
    with open(f"{PROJECT}/v14/their_frag_uni.pkl", "rb") as f:
        their_frags = pickle.load(f)
    bde_pre = {}
    for _reg in ("pos", "neg", "na"):
        try:
            with open(f"{PROJECT}/data/bde_frag_{_reg}.pkl", "rb") as f:
                bde_pre[_reg] = pickle.load(f)
        except FileNotFoundError:
            bde_pre[_reg] = {}

    bde_pre["nh4"] = bde_pre["pos"]  # NH4+ seeks {N,O}, identical to pos

    def bde_masses(s, adduct):
        hit = bde_pre[bde_regime(adduct)].get(s)
        if hit is not None and len(hit):
            return np.asarray(hit, float)
        try:
            return np.asarray(cached_fragments_bde(s, adduct), float)
        except Exception:
            return np.zeros(0)

    tr = pd.read_parquet(
        f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "molecular_formula", "adduct", "precursor_mz"])
    smass = tr.groupby("normalized_smiles")["precursor_mz"].median().to_dict()
    _meta = pd.read_parquet(
        f"{PROJECT}/data/train.parquet", columns=["normalized_smiles", "adduct", "num_peaks"])
    _rep_smi = _meta.sort_values("num_peaks").groupby("normalized_smiles").tail(1)
    _rep_smi = _rep_smi["normalized_smiles"].tolist()
    del _meta
    gc.collect()
    _rspec = pd.read_parquet(
        f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "adduct", "ms2_mzs", "ms2_normalized_intensities"],
        filters=[("normalized_smiles", "in", _rep_smi)])
    _rspec["_m"] = _rspec["normalized_smiles"].map(smass)
    _rspec = _rspec.sort_values(["_m", "normalized_smiles"]).drop_duplicates(
        ["normalized_smiles", "adduct"], keep="last").reset_index(drop=True)
    _ml = [np.asarray(m, dtype=np.float32) for m in _rspec["ms2_mzs"]]
    _il = [np.asarray(v, dtype=np.float32) for v in _rspec["ms2_normalized_intensities"]]
    _oo = np.zeros(len(_rspec) + 1, np.int64)
    for _i, _m in enumerate(_ml):
        _oo[_i + 1] = _oo[_i] + len(_m)
    _rep_lib = {"off": _oo, "mz": np.concatenate(_ml), "it": np.concatenate(_il)}
    _rep = np.arange(len(_rspec))
    _rep_key = np.asarray(_rspec["normalized_smiles"].tolist(), dtype=object)
    _rep_nm = np.asarray(_rspec["_m"].tolist(), dtype=float)
    _rep_ad = np.asarray(_rspec["adduct"].astype(str).tolist(), dtype=object)
    del _rspec, _ml, _il
    gc.collect()

    def score_X(cands, q, qrows, target, frag_fn):
        lib_rows = _spec[_spec["normalized_smiles"].isin(set(cands))]
        lib = load_library_df(lib_rows, L.neutral_mass)
        specs = [(q["ms2_mzs"], q["ms2_normalized_intensities"], q["adduct"])]
        lib_hits = L.lib_sim(lib, specs, target)
        lv = np.array([lib_hits.get(s, 0.0) for s in cands], np.float32)
        an = L.analog_sim(_rep_lib, specs, target, _rep, _rep_key, _rep_nm, _rep_ad)
        afp, asim = [], []
        for k, v in an[:80]:
            t = F.fp_and_mass(k)
            if t is not None:
                afp.append(t[0].astype(np.float32))
                asim.append(v)
        afp = np.stack(afp) if afp else None
        asim = np.array(asim, np.float32)
        ce = q["collision_energy_ev"]
        try:
            ce = float(np.mean(np.atleast_1d(ce))) if ce is not None and len(np.atleast_1d(ce)) else 25.0
        except Exception:
            ce = 25.0
        zlog = F._logits_raw([(q["ms2_mzs"], q["ms2_normalized_intensities"])], [tnet],
                             q["precursor_mz"], q["adduct"], q["instrument_type"], ce, 1.0)
        cfp = []
        for s in cands:
            t = F.fp_and_mass(s)
            cfp.append(t[0].astype(np.float32) if t is not None else np.zeros(6930, np.float32))
        cfp = np.stack(cfp)
        fr = []
        for s in cands:
            f = frag_fn(s)
            m2 = np.asarray(q["ms2_mzs"], float)
            i2 = np.asarray(q["ms2_normalized_intensities"], float)
            md = 1.0 if str(q["adduct"]).rstrip().endswith("]+") else -1.0
            fr.append(float(their_explain(np.asarray(f, float), m2, i2,
                                          mode=md, tol=0.01)) if len(f) else 0.0)
        fr = np.array(fr, np.float32)
        X = rank_features(cfp, lv, afp, asim, model_logits=zlog, frag=fr)
        del lib
        return X

    def rank_of(pt, cands, truth):
        return next((i + 1 for i, (_, s) in enumerate(
            sorted(zip(pt, cands), reverse=True)) if s == truth), 10 ** 9)

    rows = []
    for seed in SEEDS:
        d = pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{seed}.parquet")
        groups = list(d.groupby(["seed", "truth"]))
        if n_query is not None:
            groups = groups[:n_query]
        _uni = set()
        for _, _g in groups:
            _uni.update(_g["cand"].tolist())
        _uni.update([_t for (_sd, _t), _ in groups])
        global _spec
        _spec = pd.read_parquet(
            f"{PROJECT}/data/train.parquet",
            columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz", "ms2_mzs",
                     "ms2_normalized_intensities", "instrument_type",
                     "collision_energy_ev", "ionization_mode"],
            filters=[("normalized_smiles", "in", sorted(_uni))])
        for (sd, truth), g in groups:
            cands = g["cand"].tolist()
            qrows = _spec[_spec["normalized_smiles"] == truth].head(2)
            if len(qrows) == 0:
                continue
            q = qrows.iloc[0]
            target = float(np.median([nm(p, a) for p, a in
                                      zip(qrows["precursor_mz"], qrows["adduct"])]))
            frag_fns = (("their", lambda s: np.asarray(
                their_frags.get(s, np.zeros(0)), float)),
                ("bde", lambda s: bde_masses(s, q["adduct"])))
            Xs = {fax: score_X(cands, q, qrows, target, fn) for fax, fn in frag_fns}
            R = {"old": rankers, "new": rankers_bde}
            for fax, X in Xs.items():
                for rname, rmodels in R.items():
                    pt = np.mean([m.predict_proba(X)[:, 1]
                                  for m in rmodels.values()], axis=0)
                    rt = rank_of(pt, cands, truth)
                    rows.append((seed, f"{rname}+{fax}",
                                 1 / rt if rt <= 25 else 0.0))
        for _arm in ("old+their", "old+bde", "new+their", "new+bde"):
            _a = np.mean([r for (sd, arm, r) in rows if sd == seed and arm == _arm])
            print(f"seed {seed}: {_arm}={_a:.3f}", flush=True)
        del d, _spec
        gc.collect()
    df = pd.DataFrame(rows, columns=["seed", "arm", "rr"])
    df.to_csv(f"{PROJECT}/v14/fr_bout_log.csv", index=False)
    for _arm in ("old+their", "old+bde", "new+their", "new+bde"):
        print(f"BOUT {_arm}=%.3f (n=%d)" % (
            df[df.arm == _arm].rr.mean(), (df.arm == _arm).sum()), flush=True)
    print("wrote v14/fr_bout_log.csv", flush=True)


if __name__ == "__main__":
    _n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(_n)
