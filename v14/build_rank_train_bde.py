"""Build BDE-based rank_train: same protocol as THEIR npz, BDE frag masses.
819 train queries (bout truths excluded), +-8.5ppm train+COCONUT windows,
half class-1 (M=0, library keeps same-structure), half class-2 (M=1, dropped).
Same v14.rank.rank_features code as inference. -> data/rank_train_bde.npz
Usage: .venv/bin/python -m v14.build_rank_train_bde
"""
import gc
import pickle

import numpy as np
import pandas as pd
import torch

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
N_Q = 819
PPM = 8.5
SEED = 7


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


def main():
    import v14.libsearch as L
    from v14.loaders import load_library_df
    from v14.rank import rank_features
    import v13.fork_fpnet_raw as F
    from v13.fork_frag_raw import explain_score as their_explain
    from v13.frag_up import cached_fragments_bde
    from v13.frag_up import _regime as bde_regime
    from bisect import bisect_left, bisect_right

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

    bde_pre = {}
    for _reg in ("pos", "neg", "na"):
        with open(f"{PROJECT}/data/bde_frag_{_reg}.pkl", "rb") as f:
            bde_pre[_reg] = pickle.load(f)
    bde_pre["nh4"] = bde_pre["pos"]
    _fp_cache, _bde_cache = {}, {}

    def fp_of(s):
        v = _fp_cache.get(s)
        if v is None:
            t = F.fp_and_mass(s)
            v = t[0].astype(np.float32) if t is not None else np.zeros(6930, np.float32)
            _fp_cache[s] = v
        return v

    def bde_of(s, adduct):
        k = (s, bde_regime(adduct))
        v = _bde_cache.get(k)
        if v is None:
            hit = bde_pre[k[1]].get(s)
            if hit is not None and len(hit):
                v = np.asarray(hit, float)
            else:
                try:
                    v = np.asarray(cached_fragments_bde(s, adduct), float)
                except Exception:
                    v = np.zeros(0)
            _bde_cache[k] = v
        return v

    banned = set()
    for sd in (10, 11, 12):
        d = pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{sd}.parquet")
        banned.update(d["truth"].tolist())
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet",
                         columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                                  "ms2_mzs", "ms2_normalized_intensities", "num_peaks",
                                  "instrument_type", "collision_energy_ev"])
    tr["_neut"] = [nm(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    tr = tr[np.isfinite(tr["_neut"].values)]
    qpool = [s for s in tr["normalized_smiles"].unique() if s not in banned]
    rng = np.random.default_rng(SEED)
    queries = list(rng.choice(sorted(qpool), size=N_Q, replace=False))
    tstruct = tr.groupby("normalized_smiles")["_neut"].median()
    tmass = tstruct.sort_values().values
    tsmi = tstruct.sort_values().index.values
    cf = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet",
                         columns=["canonical_smiles", "exact_molecular_weight"])
    co = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    del cf
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values
    # global analog rep pool (full train, static)
    _meta = tr[["normalized_smiles", "adduct", "num_peaks"]].copy()
    _rep_smi = _meta.sort_values("num_peaks").groupby("normalized_smiles").tail(1)
    _rep_smi = _rep_smi["normalized_smiles"].tolist()
    _rspec = tr[tr["normalized_smiles"].isin(set(_rep_smi))][
        ["normalized_smiles", "adduct", "ms2_mzs", "ms2_normalized_intensities", "_neut"]]
    _rspec = _rspec.sort_values(["_neut", "normalized_smiles"]).drop_duplicates(
        ["normalized_smiles", "adduct"], keep="last").reset_index(drop=True)
    _ml = [np.asarray(m, dtype=np.float32) for m in _rspec["ms2_mzs"]]
    _il = [np.asarray(v, dtype=np.float32) for v in _rspec["ms2_normalized_intensities"]]
    _oo = np.zeros(len(_rspec) + 1, np.int64)
    for _i, _m in enumerate(_ml):
        _oo[_i + 1] = _oo[_i] + len(_m)
    _rep_lib = {"off": _oo, "mz": np.concatenate(_ml), "it": np.concatenate(_il)}
    _rep = np.arange(len(_rspec))
    _rep_key = np.asarray(_rspec["normalized_smiles"].tolist(), dtype=object)
    _rep_nm = np.asarray(_rspec["_neut"].tolist(), dtype=float)
    _rep_ad = np.asarray(_rspec["adduct"].astype(str).tolist(), dtype=object)
    del _rspec, _ml, _il
    gc.collect()

    Xs, Ys, Ms, Gs = [], [], [], []
    for qi, qs in enumerate(queries):
        cls1 = (qi % 2 == 0)
        qrows = tr[tr["normalized_smiles"] == qs]
        qrows = qrows.sort_values("num_peaks", ascending=False)
        q = qrows.iloc[0]
        target = float(qrows["_neut"].median())
        tol = target * PPM / 1e6
        lo = bisect_left(tmass, target - tol)
        hi = bisect_right(tmass, target + tol)
        cands = list(tsmi[lo:hi])
        lo2 = bisect_left(cmass, target - tol)
        hi2 = bisect_right(cmass, target + tol)
        cands += list(csmi[lo2:hi2])
        cands = list(dict.fromkeys(cands))
        if qs not in cands:
            cands.append(qs)
        lib_tr = tr if cls1 else tr[tr["normalized_smiles"] != qs]
        lib = load_library_df(
            lib_tr[lib_tr["normalized_smiles"].isin(set(cands))], L.neutral_mass)
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
        cfp = np.stack([fp_of(s) for s in cands])
        fr = []
        for s in cands:
            f = bde_of(s, q["adduct"])
            m2 = np.asarray(q["ms2_mzs"], float)
            i2 = np.asarray(q["ms2_normalized_intensities"], float)
            md = 1.0 if str(q["adduct"]).rstrip().endswith("]+") else -1.0
            fr.append(float(their_explain(np.asarray(f, float), m2, i2,
                                          mode=md, tol=0.01)) if len(f) else 0.0)
        fr = np.array(fr, np.float32)
        X = rank_features(cfp, lv, afp, asim, model_logits=zlog, frag=fr)
        Xs.append(X)
        Ys.append(np.array([1.0 if s == qs else 0.0 for s in cands]))
        Ms.append(np.full(len(cands), 0.0 if cls1 else 1.0))
        Gs.append(np.full(len(cands), qi, dtype=np.int64))
        del lib
        gc.collect()
        if (qi + 1) % 100 == 0:
            print(f"{qi + 1}/{len(queries)} rows={sum(map(len, Xs))} "
                  f"fp_cache={len(_fp_cache)} bde_cache={len(_bde_cache)}", flush=True)
    X = np.vstack(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float64)
    M = np.concatenate(Ms).astype(np.float64)
    G = np.concatenate(Gs).astype(np.int64)
    np.savez(f"{PROJECT}/data/rank_train_bde.npz", X=X, Y=Y, M=M, G=G)
    print(f"saved rows={len(X)} groups={len(np.unique(G))} pos={Y.mean():.4f} "
          f"M0={(M == 0).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
