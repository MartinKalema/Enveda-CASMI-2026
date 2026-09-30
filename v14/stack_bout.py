"""Stack-vs-stack: THEIR ranker (31 feats, trained weights) vs OUR GBM-9.
Identical frozen queries (seeds 10/11/12 from v5 cache). Per-candidate
features computed with each side's own machinery. MRR@25 both.
Usage: .venv/bin/python -m v14.stack_bout --stage {prefrag,bout}
"""
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
from bisect import bisect_left, bisect_right

from v1.subformula import ADDUCT_DELTA

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)


def nm(p, a):
    if a == "[2M+H]+":
        return (p - 1.007276) / 2
    if a == "[2M+Na]+":
        return (p - 22.989218) / 2
    if a == "[2M-H]-":
        return (p + 1.007276) / 2
    d = ADDUCT_DELTA.get(a)
    return p - d if d is not None else np.nan


def stage_prefrag():
    """Precompute their frag masses for the candidate universe (all seeds)."""
    import v13.fork_frag_raw as FF
    uni = set()
    for seed in SEEDS:
        d = pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{seed}.parquet")
        uni.update(d["cand"].unique().tolist())
    uni = sorted(uni)
    print(f"universe: {len(uni)}", flush=True)
    out = {}
    for i, s in enumerate(uni):
        try:
            out[s] = np.asarray(FF._frag_masses_wrapper(s), dtype=float)
        except Exception:
            out[s] = np.zeros(0)
        if (i + 1) % 2000 == 0:
            print(f"frag {i + 1}/{len(uni)}", flush=True)
    with open(f"{PROJECT}/v14/their_frag_uni.pkl", "wb") as f:
        pickle.dump(out, f)
    print("saved their_frag_uni.pkl", flush=True)


def stage_bout(n_query=None, seeds=SEEDS):
    import v14.libsearch as L
    from v14.loaders import load_library_df
    from v14.rank import rank_features
    import v13.fork_fpnet_raw as F
    from v2.train_fp import FpMLP, bin_spectrum, meta_vec, N_BINS, ADDUCTS
    from v13.frag_up import cached_fragments_bde
    from v2.blend import cosine, tanimoto
    from v4.channels import entropy_similarity

    device = "cpu"
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
    mynet = FpMLP(d_h=1536).to(device)
    mynet.load_state_dict(torch.load(f"{PROJECT}/data/fp_trans.pt", map_location=device))
    mynet.eval()
    rankers = pickle.load(open(f"{PROJECT}/v13/their_ranker.pkl", "rb"))
    gbm9 = pickle.load(open(f"{PROJECT}/v13/gbm9_keyed.pkl", "rb"))
    tf = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    scol = "normalized_smiles" if "normalized_smiles" in tf.columns else "smiles"
    myfp = {s: np.unpackbits(np.asarray(x, dtype=np.uint8)).astype(np.float32)
            for s, x in zip(tf[scol], tf["fp"])}
    cf = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet", columns=["canonical_smiles", "fp"])
    for s, x in zip(cf["canonical_smiles"], cf["fp"]):
        myfp.setdefault(s, np.unpackbits(np.asarray(x, dtype=np.uint8)).astype(np.float32))
    with open(f"{PROJECT}/v14/their_frag_uni.pkl", "rb") as f:
        their_frags = pickle.load(f)
    with open(f"{PROJECT}/data/frag_cache.pkl", "rb") as f:
        our_frags = pickle.load(f)
    from v7.submit_v7 import frag_match
    from v13.enrich_bde import ladder_score
    _tm = tr.groupby("normalized_smiles")["neutral"].median() if "neutral" in tr.columns else None
    tr["_neut"] = [nm(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    smass = tr.groupby("normalized_smiles")["_neut"].median().to_dict()
    sform = tr.groupby("normalized_smiles")["molecular_formula"].first().to_dict()
    _cfm = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet",
                           columns=["canonical_smiles", "molecular_formula", "exact_molecular_weight"])
    for _s, _m, _ff in zip(_cfm["canonical_smiles"], _cfm["exact_molecular_weight"],
                           _cfm["molecular_formula"]):
        smass.setdefault(_s, float(_m))
        sform.setdefault(_s, _ff)
    from collections import Counter as _Counter
    fprior = _Counter(tr["molecular_formula"].tolist())
    gbm9m, gbm9f = gbm9["model"], gbm9["feats"]
    tr = pd.read_parquet(
        f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "molecular_formula", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities", "instrument_type",
                 "collision_energy_ev", "ionization_mode"])
    res = {"theirs": [], "ours": []}
    for seed in seeds:
        d = pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{seed}.parquet")
        groups = list(d.groupby(["seed", "truth"]))
        if n_query is not None:
            groups = groups[:n_query]
        for (sd, truth), g in groups:
            cands = g["cand"].tolist()
            qrows = tr[tr["normalized_smiles"] == truth].head(2)
            if len(qrows) == 0:
                continue
            q = qrows.iloc[0]
            # ---- THEIR side ----
            lib_rows = tr[tr["normalized_smiles"].isin(set(cands))]
            lib = load_library_df(lib_rows, L.neutral_mass)
            specs = [(q["ms2_mzs"], q["ms2_normalized_intensities"], q["adduct"])]
            target = float(np.median([nm(p, a) for p, a in
                                      zip(qrows["precursor_mz"], qrows["adduct"])]))
            lib_hits = L.lib_sim(lib, specs, target)
            lv = np.array([lib_hits.get(s, 0.0) for s in cands], np.float32)
            rep, rep_key, rep_nm, rep_ad = L.build_rep(lib)
            an = L.analog_sim(lib, specs, target, rep, rep_key, rep_nm, rep_ad)
            amd = {k: v for k, v in an}
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
                f = their_frags.get(s, np.zeros(0))
                m2 = np.asarray(q["ms2_mzs"], float)
                fr.append(float((np.isin(np.round(m2 / 0.01).astype(int),
                                         np.round(f / 0.01).astype(int))).mean()) if len(f) else 0.0)
            fr = np.array(fr, np.float32)
            X = rank_features(cfp, lv, afp, asim, model_logits=zlog, frag=fr)
            pt = np.mean([m.predict_proba(X)[:, 1] for m in rankers.values()], axis=0)
            rt = next((i + 1 for i, (_, s) in enumerate(
                sorted(zip(pt, cands), reverse=True)) if s == truth), 10 ** 9)
            res["theirs"].append(1 / rt if rt <= 25 else 0.0)
            # ---- OUR side: GBM-9 over 9 features ----
            from v2.train_fp import bin_spectrum, meta_vec, N_BINS, ADDUCTS
            from v13.enrich_bde import ladder_score
            qmass = target
            _ff = []
            for _, r in qrows.iterrows():
                _v = np.zeros(N_BINS + len(ADDUCTS) + 2, dtype=np.float32)
                _v[:N_BINS] = bin_spectrum(r["ms2_mzs"], r["ms2_normalized_intensities"])
                _v[N_BINS:] = meta_vec(r["adduct"], r["precursor_mz"], r["collision_energy_ev"])
                _ff.append(_v)
            with torch.no_grad():
                _Z = mynet(torch.from_numpy(np.stack(_ff))).numpy().mean(axis=0)
            Zn = _Z / (np.linalg.norm(_Z) + 1e-9)
            _top = g.sort_values("ana", ascending=False)["cand"].head(3).tolist()
            _tv = [myfp[t] for t in _top if t in myfp]
            _feats, _order = [], []
            for _, c in g.iterrows():
                f = myfp.get(c["cand"])
                _tt1 = 0.0
                if f is not None and _tv:
                    _fb = f > 0.5
                    _tt1 = max(float(np.logical_and(_fb, _t > 0.5).sum()) /
                               float(np.logical_or(_fb, _t > 0.5).sum()) for _t in _tv)
                _mf = 0.0
                for _, r in qrows.iterrows():
                    _v = frag_match(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                    our_frags.get(c["cand"], []), r["adduct"])
                    if _v > _mf:
                        _mf = _v
                _bd = 0.0
                for _, r in qrows.iterrows():
                    _v = ladder_score(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                      cached_fragments_bde(c["cand"], r["adduct"]))
                    if _v > _bd:
                        _bd = _v
                _cm = smass.get(c["cand"], np.nan)
                _fd = 0.0
                if f is not None:
                    _fn = f / (np.linalg.norm(f) + 1e-9)
                    _fd = float(_fn @ Zn)
                _feats.append([c["cos"], c["ent"], c["ana"],
                               abs(_cm - qmass) / qmass if _cm == _cm else 1.0,
                               float(np.log1p(fprior.get(sform.get(c["cand"], ""), 0))),
                               _tt1, _mf, _fd, _bd])
                _order.append(c["cand"])
            _po = gbm9m.predict_proba(np.array(_feats, dtype=np.float32))[:, 1]
            ro = sorted(zip(_po, _order), reverse=True)
            ro_rank = next((i + 1 for i, (_, s) in enumerate(ro) if s == truth), 10 ** 9)
            res["ours"].append(1 / ro_rank if ro_rank <= 25 else 0.0)
        print(f"seed {seed}: theirs={np.mean(res['theirs'][-50:]):.3f} "
              f"ours={np.mean(res['ours'][-50:]):.3f}", flush=True)
    print("BOUT: theirs=%.3f ours=%.3f (n=%d)" % (
        np.mean(res["theirs"]), np.mean(res["ours"]), len(res["theirs"])))


if __name__ == "__main__":
    import sys
    if "--stage prefrag" in sys.argv:
        stage_prefrag()
    else:
        stage_bout()
