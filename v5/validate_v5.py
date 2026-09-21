"""v5 calibrated validation: all channels, 3 seeds, anchor-checked.
Caches per-query candidate features to v5/feat_cache/ for mix evaluation.
Channels: cos (v0 anchor), ent, ana (analog prop), exp (explained-intensity).
"""
import numpy as np, pandas as pd, os
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA, explained_intensity
from v2.blend import cosine, tanimoto
from v4.channels import entropy_similarity

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
CACHE = f"{PROJECT}/v5/feat_cache"
P_POW, DM_MAX = 3.0, 200.0
N_Q, SEEDS = 50, (10, 11, 12)


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def window(masses, smi, qmass, ppm=20, min_n=200, cap=800):
    tol = qmass * ppm / 1e6
    lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
    pcur = ppm
    while hi - lo < min_n and pcur < 500:
        pcur *= 2; tol = qmass * pcur / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
    return list(smi[lo:hi][:cap])


def build_seed(seed):
    os.makedirs(CACHE, exist_ok=True)
    out_path = f"{CACHE}/seed{seed}.parquet"
    if os.path.exists(out_path):
        print(f"seed {seed}: cache hit", flush=True)
        return pd.read_parquet(out_path)
    rng = np.random.default_rng(seed)
    cf = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet")
    cfp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8)) for s, f in zip(cf["canonical_smiles"], cf["fp"])}
    tf = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    scol = "normalized_smiles" if "normalized_smiles" in tf.columns else "smiles"
    tfp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8)) for s, f in zip(tf[scol], tf["fp"])}
    train = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "molecular_formula", "adduct",
                 "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    groups = np.array(train["inchikey14"].unique())
    held = set(rng.choice(groups, size=min(800, len(groups) // 12), replace=False))
    qpool = train[train["inchikey14"].isin(held)]
    structs = np.array(qpool["normalized_smiles"].unique())
    rng.shuffle(structs)
    queries = list(structs[:N_Q])
    db = train[~train["inchikey14"].isin(held)]
    struct = db.groupby("normalized_smiles")["neutral"].median().reset_index()
    truth = qpool.groupby("normalized_smiles")["neutral"].median().reset_index()
    struct = pd.concat([struct, truth]).drop_duplicates("normalized_smiles")
    struct = struct.sort_values("neutral").reset_index(drop=True)
    masses = struct["neutral"].values
    smi = struct["normalized_smiles"].values
    fmap = dict(zip(db["normalized_smiles"], db["molecular_formula"]))
    fmap.update(dict(zip(truth["normalized_smiles"],
                         qpool.groupby("normalized_smiles")["molecular_formula"].first())))
    db_samp = db.groupby("normalized_smiles").head(2).reset_index(drop=True)
    tneut = db_samp["neutral"].values
    co = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values

    rows = []
    for qi, qs in enumerate(queries):
        qspec = qpool[qpool["normalized_smiles"] == qs]
        qmass = float(np.median([neutral_mass(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        tcands = window(masses, smi, qmass)
        ccands = window(cmass, csmi, qmass, cap=1200)
        twin = db_samp[db_samp["normalized_smiles"].isin(set(tcands))]
        espec, cspec = {}, {}
        for _, r in qspec.iterrows():
            for _, t in twin.iterrows():
                e = entropy_similarity(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                       t["ms2_mzs"], t["ms2_normalized_intensities"])
                c = cosine(r["ms2_mzs"], r["ms2_normalized_intensities"],
                           t["ms2_mzs"], t["ms2_normalized_intensities"])
                k = t["normalized_smiles"]
                if e > espec.get(k, 0): espec[k] = e
                if c > cspec.get(k, 0): cspec[k] = c
        dm = np.abs(tneut - qmass)
        pool = db_samp[dm <= DM_MAX]
        if len(pool) > 1200: pool = pool.sample(1200, random_state=qi)
        analogs = []
        for _, t in pool.iterrows():
            best = 0.0
            for _, r in qspec.iterrows():
                e = entropy_similarity(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                       t["ms2_mzs"], t["ms2_normalized_intensities"])
                if e > best: best = e
            if best > 0.05: analogs.append((best, t["normalized_smiles"]))
        analogs.sort(reverse=True)
        analogs = analogs[:40]
        afps = [(s, tfp[s]) for _, s in analogs if s in tfp]
        asims = [a[0] for a in analogs[:len(afps)]]
        for s in tcands + ccands:
            f = tfp.get(s, cfp.get(s))
            ab = 0.0
            if f is not None:
                for sim, (_, g) in zip(asims, afps):
                    v = (sim ** P_POW) * tanimoto(f, g)
                    if v > ab: ab = v
            rows.append((seed, qs, s, cspec.get(s, 0.0), espec.get(s, 0.0), ab))
        if (qi + 1) % 10 == 0: print(f"seed {seed}: {qi+1}/{N_Q}", flush=True)
    feat = pd.DataFrame(rows, columns=["seed", "truth", "cand", "cos", "ent", "ana"])
    feat.to_parquet(out_path)
    return feat


def evaluate():
    import glob as _g
    frames = []
    for seed in SEEDS:
        frames.append(build_seed(seed))
    feat = pd.concat(frames)
    mixes = {
        "cos": lambda d: d["cos"],
        "ent": lambda d: d["ent"],
        "ana": lambda d: d["ana"],
        "max(cos,ent)": lambda d: np.maximum(d["cos"], d["ent"]),
        "max(all)": lambda d: np.maximum.reduce([d["cos"], d["ent"], d["ana"]]),
    }
    print("mix | seed MRRs | mean", flush=True)
    for name, fn in mixes.items():
        per_seed = []
        for seed in SEEDS:
            d = feat[feat["seed"] == seed].copy()
            d["s"] = fn(d)
            rr = []
            for truth, g in d.groupby("truth"):
                g = g.sort_values("s", ascending=False)
                rank = next((i + 1 for i, (_, r) in enumerate(g.iterrows()) if r["cand"] == truth), 10**9)
                rr.append(1 / rank if rank <= 25 else 0.0)
            per_seed.append(float(np.mean(rr)))
        print(f"{name:12s} " + " ".join(f"{v:.3f}" for v in per_seed) + f" mean={np.mean(per_seed):.3f}", flush=True)


if __name__ == "__main__":
    evaluate()
