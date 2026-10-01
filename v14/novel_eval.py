"""Novel-molecule end-to-end: held-out inchikey14 groups, retrieval included.
Same stack as production (old ranker, BDE frags, their scorer); ONLY the
COCONUT pool file varies (old 627k vs orig 739k). Identical queries both arms.
Usage: .venv/bin/python -m v14.novel_eval [n_held]
Writes v14/novel_eval_log.csv (qid, arm, truth_in_pool, rr).
"""
import gc
import pickle
import sys
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd
import torch

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
N_HELD = 60
SEED = 21
_ik_cache = {}


def ik14(smi):
    """First-block InChIKey for ANY smiles string (cross-canonicalization identity)."""
    v = _ik_cache.get(smi)
    if v is None:
        try:
            from rdkit import Chem
            m = Chem.MolFromSmiles(smi)
            v = Chem.MolToInchiKey(m).split("-")[0] if m is not None else ""
        except Exception:
            v = ""
        _ik_cache[smi] = v
    return v


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


def main(n_held=N_HELD):
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
    bde_old, bde_new = {}, {}
    for _reg in ("pos", "neg", "na"):
        with open(f"{PROJECT}/data/bde_frag_{_reg}.pkl", "rb") as f:
            bde_old[_reg] = pickle.load(f)
        with open(f"{PROJECT}/data/orig_bde_{_reg}.pkl", "rb") as f:
            bde_new[_reg] = pickle.load(f)
    for _d in (bde_old, bde_new):
        _d["nh4"] = _d["pos"]
    _fp_cache, _bde_cache = {}, {}

    def fp_of(s):
        v = _fp_cache.get(s)
        if v is None:
            t = F.fp_and_mass(s)
            v = t[0].astype(np.float32) if t is not None else np.zeros(6930, np.float32)
            _fp_cache[s] = v
        return v

    def bde_of(s, adduct, table):
        k = (s, bde_regime(adduct))
        v = _bde_cache.get((k, id(table)))
        if v is None:
            hit = table[k[1]].get(s)
            if hit is not None and len(hit):
                v = np.asarray(hit, float)
            else:
                try:
                    v = np.asarray(cached_fragments_bde(s, adduct), float)
                except Exception:
                    v = np.zeros(0)
            _bde_cache[(k, id(table))] = v
        return v

    tr = pd.read_parquet(
        f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities", "num_peaks",
                 "instrument_type", "collision_energy_ev"])
    tr["_neut"] = [nm(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    tr = tr[np.isfinite(tr["_neut"].values)]
    rng = np.random.default_rng(SEED)
    groups = sorted(tr["inchikey14"].unique())
    held = set(rng.choice(groups, size=n_held, replace=False))
    lib_tr = tr[~tr["inchikey14"].isin(held)].reset_index(drop=True)
    tstruct = lib_tr.groupby("normalized_smiles")["_neut"].median()
    tmass = tstruct.sort_values().values
    tsmi = tstruct.sort_values().index.values
    pools = {}
    for name, path in (("old", f"{PROJECT}/data/coconut_fp.parquet"),
                       ("new", f"{PROJECT}/data/coconut_orig.parquet")):
        cf = pd.read_parquet(path, columns=["canonical_smiles", "exact_molecular_weight"])
        co = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
        del cf
        pools[name] = (co["exact_molecular_weight"].values, co["canonical_smiles"].values)
    # cross-canonicalization identity: query truth ik (first block)
    qik_of = dict(tr.groupby("normalized_smiles")["inchikey14"].first())
    # orig-pool smiles -> ik dict from the source CSV (train pool resolved via RDKit cache)
    orig_ik = {}
    for ch in pd.read_csv("/tmp/coconut-orig/coconut_csv_lite-10-2026.csv",
                          usecols=["canonical_smiles", "standard_inchi_key"],
                          chunksize=200000):
        for s, k in zip(ch["canonical_smiles"], ch["standard_inchi_key"]):
            if s not in orig_ik:
                orig_ik[s] = str(k).split("-")[0] if k == k else ""
    # analog reps from visible library only (held structures excluded)
    _meta = lib_tr[["normalized_smiles", "adduct", "num_peaks"]].copy()
    _rep_smi = _meta.sort_values("num_peaks").groupby("normalized_smiles").tail(1)
    _rep_smi = _rep_smi["normalized_smiles"].tolist()
    _rspec = lib_tr[lib_tr["normalized_smiles"].isin(set(_rep_smi))][
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

    rows = []
    held_q = tr[tr["inchikey14"].isin(held)].groupby("inchikey14")
    for hi, (ik, qspec) in enumerate(held_q):
        q = qspec.iloc[0]
        qs = q["normalized_smiles"]
        truth_ik = qik_of.get(qs, "")
        target = float(qspec["_neut"].median())
        specs = [(q["ms2_mzs"], q["ms2_normalized_intensities"], q["adduct"])]
        for arm, (cmass, csmi) in pools.items():
            cands = list(dict.fromkeys(
                window10(tmass, tsmi, target) + window10(cmass, csmi, target)))
            if arm == "new":
                cand_ik = [orig_ik.get(s, ik14(s)) for s in cands]
            else:
                cand_ik = [qik_of.get(s, ik14(s)) for s in cands]
            in_pool = truth_ik != "" and truth_ik in cand_ik
            lib = load_library_df(
                lib_tr[lib_tr["normalized_smiles"].isin(set(cands))], L.neutral_mass)
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
            table = bde_old if arm == "old" else bde_new
            fr = []
            for s in cands:
                f = bde_of(s, q["adduct"], table)
                m2 = np.asarray(q["ms2_mzs"], float)
                i2 = np.asarray(q["ms2_normalized_intensities"], float)
                md = 1.0 if str(q["adduct"]).rstrip().endswith("]+") else -1.0
                fr.append(float(their_explain(np.asarray(f, float), m2, i2,
                                              mode=md, tol=0.01)) if len(f) else 0.0)
            fr = np.array(fr, np.float32)
            X = rank_features(cfp, lv, afp, asim, model_logits=zlog, frag=fr)
            pt = np.mean([m.predict_proba(X)[:, 1] for m in rankers.values()], axis=0)
            rt = next((i + 1 for i, (_, s, k) in enumerate(
                sorted(zip(pt, cands, cand_ik), reverse=True)) if k == truth_ik),
                10 ** 9) if truth_ik else 10 ** 9
            rows.append((ik, arm, int(in_pool), 1 / rt if rt <= 25 else 0.0, len(cands)))
            del lib
            gc.collect()
        if (hi + 1) % 15 == 0:
            print(f"{hi + 1} done", flush=True)
    df = pd.DataFrame(rows, columns=["qid", "arm", "in_pool", "rr", "nc"])
    df.to_csv(f"{PROJECT}/v14/novel_eval_log.csv", index=False)
    for arm in ("old", "new"):
        d = df[df.arm == arm]
        print(f"NOVEL {arm}: recall={d.in_pool.mean():.3f} "
              f"mrr={d.rr.mean():.3f} mrr|hit={d[d.in_pool == 1].rr.mean():.3f} (n={len(d)})",
              flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_HELD)
