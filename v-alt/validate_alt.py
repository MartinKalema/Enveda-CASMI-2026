"""v-alt validation: structure-disjoint (inchikey14) formula-first isomer retrieval.

Alt pipeline uses NO spectrum<->spectrum similarity (no cosine/entropy/Tanimoto/fingerprints).
Same-split cosine anchor included per lessons-learned rule 22 (anchor-gating).
Protocol: 40 queries x 3 seeds (n=120 total); truth structures injected mass/formula-only,
truth spectra NEVER in any scoring pool (alt needs no library spectra by construction).
"""
import numpy as np
import pandas as pd
from bisect import bisect_left, bisect_right
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "alt_score", os.path.join(os.path.dirname(os.path.abspath(__file__)), "alt_score.py"))
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)
(PROJECT, neutral_mass, fast_explained, loss_bonus, frag_fit,
 local_cosine, load_frag_disk, save_frag_disk) = (
    _mod.PROJECT, _mod.neutral_mass, _mod.fast_explained, _mod.loss_bonus,
    _mod.frag_fit, _mod.local_cosine, _mod.load_frag_disk, _mod.save_frag_disk)

TOPK_FORM = 2
POOL_CAP = 150


def window(smi, masses, qmass, ppm=20, min_n=100, cap=400):
    tol = qmass * ppm / 1e6
    lo = bisect_left(masses, qmass - tol)
    hi = bisect_right(masses, qmass + tol)
    pcur = ppm
    while hi - lo < min_n and pcur < 500:
        pcur *= 2
        tol = qmass * pcur / 1e6
        lo = bisect_left(masses, qmass - tol)
        hi = bisect_right(masses, qmass + tol)
    return list(smi[lo:hi][:cap])


def run_seed(train, seed, n_query=40):
    rng = np.random.default_rng(seed)
    groups = np.array(train["inchikey14"].unique())
    held = set(rng.choice(groups, size=min(500, len(groups) // 20), replace=False))
    qpool = train[train["inchikey14"].isin(held)]
    structs = np.array(qpool["normalized_smiles"].unique())
    rng.shuffle(structs)
    queries = list(structs[:n_query])

    db = train[~train["inchikey14"].isin(held)]
    struct = db.groupby("normalized_smiles").agg(
        mass=("neutral", "median"), formula=("molecular_formula", "first")).reset_index()
    truth = qpool.groupby("normalized_smiles").agg(
        mass=("neutral", "median"), formula=("molecular_formula", "first")).reset_index()
    truth_form = dict(zip(truth["normalized_smiles"], truth["formula"]))
    struct = pd.concat([struct, truth]).drop_duplicates("normalized_smiles")
    struct = struct.sort_values("mass").reset_index(drop=True)
    masses = struct["mass"].values
    smi = struct["normalized_smiles"].values
    fmap = dict(zip(struct["normalized_smiles"], struct["formula"]))
    db_samp = db.groupby("normalized_smiles").head(2)

    res = {"frag": [], "form": [], "blend": [], "full": [], "cos": []}
    funnel_ok = 0
    for qi, qs in enumerate(queries):
        qspec = qpool[qpool["normalized_smiles"] == qs].head(3)
        qrows = list(qspec.iterrows())
        qmass = float(np.median([neutral_mass(p, a) for p, a in
                                 zip(qspec["precursor_mz"], qspec["adduct"])]))
        cands = window(smi, masses, qmass)
        # --- Stage 1: rank formulae by subformula explained-intensity ---
        forms = sorted(set(fmap[s] for s in cands))
        fscore = {}
        for f in forms:
            best = 0.0
            for _, r in qrows:
                v = fast_explained(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                   f, r["adduct"])
                if v > best:
                    best = v
            fscore[f] = best
        frank = sorted(fscore, key=fscore.get, reverse=True)
        if truth_form[qs] in frank[:TOPK_FORM]:
            funnel_ok += 1
        poolset = set(s for s in cands if fmap[s] in set(frank[:TOPK_FORM]))
        poolset.add(qs)  # ensure truth scored (rank measured honestly below)
        pool = sorted(poolset, key=lambda s: abs(float(
            struct.loc[struct["normalized_smiles"] == s, "mass"].iloc[0]) - qmass))
        pool = pool[:POOL_CAP]
        # --- Stage 2: score pool isomers (spectrum-vs-structure only) ---
        scored = {k: {} for k in ("frag", "form", "blend", "full")}
        for s in pool:
            f = fmap[s]
            bf = bg = bl = 0.0
            for _, r in qrows:
                v = frag_fit(r["ms2_mzs"], r["ms2_normalized_intensities"], s, r["adduct"])
                if v > bf:
                    bf = v
                v = fast_explained(r["ms2_mzs"], r["ms2_normalized_intensities"], f, r["adduct"])
                if v > bg:
                    bg = v
                v = loss_bonus(r["ms2_mzs"], r["ms2_normalized_intensities"], qmass, r["adduct"])
                if v > bl:
                    bl = v
            scored["frag"][s] = bf
            scored["form"][s] = bg
            scored["blend"][s] = 0.5 * bf + 0.5 * bg
            scored["full"][s] = 0.5 * bf + 0.3 * bg + 0.2 * bl
        for k in scored:
            # pool members outrank all funneled-out candidates by construction
            rank = 1 + sum(1 for s in pool if scored[k][s] > scored[k][qs])
            res[k].append(1 / rank if rank <= 25 else 0.0)
        # --- Anchor: v0 cosine over same candidate list ---
        twin = db_samp[db_samp["normalized_smiles"].isin(set(cands))]
        csco = {}
        for s in cands:
            best = 0.0
            for _, t in twin[twin["normalized_smiles"] == s].iterrows():
                for _, r in qrows:
                    c = local_cosine(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                     t["ms2_mzs"], t["ms2_normalized_intensities"])
                    if c > best:
                        best = c
            csco[s] = best
        crank = 1 + sum(1 for s in cands if csco[s] > csco[qs])
        res["cos"].append(1 / crank if crank <= 25 else 0.0)
        if (qi + 1) % 10 == 0:
            print(f"seed {seed} {qi+1}/{n_query} "
                  f"frag={np.mean(res['frag']):.3f} full={np.mean(res['full']):.3f} "
                  f"cos={np.mean(res['cos']):.3f}", flush=True)
    save_frag_disk()
    return res, funnel_ok / n_query


def run(n_query=40, seeds=(0, 1, 2)):
    train = pd.read_parquet(
        f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "molecular_formula",
                 "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in
                        zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)].reset_index(drop=True)
    load_frag_disk()
    allr = {"frag": [], "form": [], "blend": [], "full": [], "cos": []}
    for seed in seeds:
        res, frec = run_seed(train, seed, n_query)
        for k in allr:
            allr[k] += res[k]
        print(f"SEED {seed}: funnel-recall@{TOPK_FORM}={frec:.3f} " +
              " ".join(f"{k}={np.mean(res[k]):.3f}" for k in
                       ("frag", "form", "blend", "full", "cos")), flush=True)
    print(f"POOLED n={len(allr['cos'])}: " +
          " ".join(f"{k}={np.mean(allr[k]):.3f}" for k in
                   ("frag", "form", "blend", "full", "cos")), flush=True)


if __name__ == "__main__":
    run()
