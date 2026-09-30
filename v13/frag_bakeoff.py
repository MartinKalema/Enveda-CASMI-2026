"""Fragmentation bake-off harness.
Contestants (each: fn(qmz, qit, frags, adduct, prec) -> score):
  F1 original  : fork frag_scores (all-bonds enumeration + h-shifts)
  F2 bde       : BDE-ordered top-14 + charge-proximal (as shipped in fork)
  F3 bde_nl    : F2 + neutral-loss menu
  F4 rear      : F2 + rearrangement SMARTS virtual fragments (when built)
Protocol: frozen queries (n=100 x seeds 30/31/32), identical 300-cap windows,
MRR@25 + win-rate vs F1. Ship threshold: beat F1 by >=0.012 on >=2 seeds.
"""
import numpy as np
import pandas as pd
from bisect import bisect_left, bisect_right

from v1.subformula import ADDUCT_DELTA
from v13.frag_up import frag_score_bde
import v13.fork_frag_raw as FF

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def nm(p, a):
    if a == "[2M+H]+":
        return (p - 1.007276) / 2
    if a == "[2M+Na]+":
        return (p - 22.989218) / 2
    if a == "[2M-H]-":
        return (p + 1.007276) / 2
    d = ADDUCT_DELTA.get(a)
    return p - d if d is not None else np.nan


def load_split(seed):
    rng = np.random.default_rng(seed)
    tr = pd.read_parquet(
        f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities"])
    tr["neutral"] = [nm(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    tr = tr[np.isfinite(tr["neutral"].values)]
    groups = np.array(tr["inchikey14"].unique())
    held = set(rng.choice(groups, size=500, replace=False))
    qp = tr[tr["inchikey14"].isin(held)]
    structs = np.array(qp["normalized_smiles"].unique())
    rng.shuffle(structs)
    db = tr[~tr["inchikey14"].isin(held)]
    st = db.groupby("normalized_smiles")["neutral"].median().reset_index()
    tq = qp.groupby("normalized_smiles")["neutral"].median().reset_index()
    st = pd.concat([st, tq]).drop_duplicates("normalized_smiles")
    st = st.sort_values("neutral").reset_index(drop=True)
    return qp, list(structs), st


def window_queries(qp, structs, st, n_query):
    masses = st["neutral"].values
    smi = st["normalized_smiles"].values
    out = []
    for qs in structs[:n_query]:
        qspec = qp[qp["normalized_smiles"] == qs]
        q = qspec.iloc[0]
        qmass = float(np.median([nm(p, a) for p, a in
                                 zip(qspec["precursor_mz"], qspec["adduct"])]))
        tol = qmass * 20 / 1e6
        lo = bisect_left(masses, qmass - tol)
        hi = bisect_right(masses, qmass + tol)
        out.append((qs, q, list(smi[lo:hi][:300])))
    return out


def score_f1(q, cands):
    specs = [(q["ms2_mzs"], q["ms2_normalized_intensities"])]
    return dict(zip(cands, (float(v) for v in FF.frag_scores(cands, specs, 1.0, workers=1))))


def score_f2(q, cands):
    out = {}
    for s in cands:
        a, _ = frag_score_bde(q["ms2_mzs"], q["ms2_normalized_intensities"],
                              s, q["adduct"], q["precursor_mz"])
        out[s] = a
    return out


def score_f3(q, cands):
    out = {}
    for s in cands:
        a, b = frag_score_bde(q["ms2_mzs"], q["ms2_normalized_intensities"],
                              s, q["adduct"], q["precursor_mz"])
        out[s] = a + 2 * b
    return out


SCORERS = {"F1-original": score_f1, "F2-bde": score_f2, "F3-bde-nl": score_f3}


def run(n_query=100, seeds=(30, 31, 32)):
    res = {k: [] for k in SCORERS}
    for seed in seeds:
        qp, structs, st = load_split(seed)
        for qs, q, cands in window_queries(qp, structs, st, n_query):
            truth = qs
            for name, fn in SCORERS.items():
                sc = fn(q, cands)
                rnk = sorted(sc.items(), key=lambda kv: -kv[1])
                rank = next((i + 1 for i, (s, _) in enumerate(rnk) if s == truth), 10 ** 9)
                res[name].append(1 / rank if rank <= 25 else 0.0)
        print(f"seed {seed}: " + " ".join(f"{k}={np.mean([v for v in res[k][-n_query:]]):.3f}"
                                          for k in SCORERS), flush=True)
    print("BAKE-OFF (300 queries):", flush=True)
    for k, v in res.items():
        print(f"  {k:12s} MRR={np.mean(v):.3f}", flush=True)


if __name__ == "__main__":
    run()
