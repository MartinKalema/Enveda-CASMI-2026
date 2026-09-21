"""v4 validation: entropy vs cosine, analog propagation (disjoint split, hit@k)."""
import numpy as np, pandas as pd
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v4.channels import entropy_similarity, tanimoto
from v2.blend import cosine

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def run(n_query=30, seed=3, p_pow=3.0, dm_max=200.0):
    rng = np.random.default_rng(seed)
    cf = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet")
    cfp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8))
           for s, f in zip(cf["canonical_smiles"], cf["fp"])}
    cmass = dict(zip(cf["canonical_smiles"], cf["exact_molecular_weight"]))
    tfp_df = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    tfp = {s: np.unpackbits(np.asarray(f, dtype=np.uint8))
           for s, f in zip(tfp_df["smiles"], tfp_df["fp"])}
    train = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    groups = np.array(train["inchikey14"].unique())
    held = set(rng.choice(groups, size=min(500, len(groups) // 20), replace=False))
    qpool = train[train["inchikey14"].isin(held)]
    structs = np.array(qpool["normalized_smiles"].unique())
    rng.shuffle(structs)
    queries = list(structs[:n_query])
    db = train[~train["inchikey14"].isin(held)]
    struct = db.groupby("normalized_smiles")["neutral"].median().reset_index()
    truth = qpool.groupby("normalized_smiles")["neutral"].median().reset_index()
    struct = pd.concat([struct, truth]).drop_duplicates("normalized_smiles")
    struct = struct.sort_values("neutral").reset_index(drop=True)
    masses = struct["neutral"].values
    smi = struct["normalized_smiles"].values
    db_samp = db.groupby("normalized_smiles").head(2)
    # analog pool: db spectra with mass within dm_max of query (spectra-rich analogs)
    coco = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    csmass = coco["exact_molecular_weight"].values
    csmi = coco["canonical_smiles"].values

    res = {"cos": ([], []), "ent": ([], []), "ana": ([], [])}
    for qi, qs in enumerate(queries):
        qspec = qpool[qpool["normalized_smiles"] == qs]
        qmass = float(np.median([neutral_mass(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        lo = bisect_left(masses, qmass - qmass * 20e-6); hi = bisect_right(masses, qmass + qmass * 20e-6)
        pcur = 20
        while hi - lo < 200 and pcur < 500:
            pcur *= 2
            lo = bisect_left(masses, qmass - qmass * pcur / 1e6)
            hi = bisect_right(masses, qmass + qmass * pcur / 1e6)
        cands = list(smi[lo:hi][:1500])
        win = db_samp[db_samp["normalized_smiles"].isin(set(cands))]
        # per-db-spectrum best entropy + cosine vs query spectra
        espec, cspec = {}, {}
        for _, r in qspec.iterrows():
            for _, t in win.iterrows():
                e = entropy_similarity(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                       t["ms2_mzs"], t["ms2_normalized_intensities"])
                c = cosine(r["ms2_mzs"], r["ms2_normalized_intensities"],
                           t["ms2_mzs"], t["ms2_normalized_intensities"])
                k = t["normalized_smiles"]
                if e > espec.get(k, 0): espec[k] = e
                if c > cspec.get(k, 0): cspec[k] = c
        # analogs: db spectra within dm_max Da (any mass), scored by entropy
        # mass prefilter (vectorized) + cap for speed
        dm = np.abs(db_samp["neutral"].values - qmass)
        pool = db_samp[dm <= dm_max]
        if len(pool) > 2000:
            pool = pool.sample(2000, random_state=qi)
        analogs = []
        for _, t in pool.iterrows():
            best = 0.0
            for _, r in qspec.iterrows():
                e = entropy_similarity(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                       t["ms2_mzs"], t["ms2_normalized_intensities"])
                if e > best: best = e
            if best > 0.05: analogs.append((best, t["normalized_smiles"]))
        analogs.sort(reverse=True)
        analogs = analogs[:50]
        afp = [(s, tfp[s], tfp.get(s)) for s, _ in [(a[1], None) for a in analogs] if s in tfp]
        # candidate scores
        clo = bisect_left(csmass, qmass - qmass * 20e-6); chi = bisect_right(csmass, qmass + qmass * 20e-6)
        cc = list(csmi[clo:chi][:1500])
        scores = {"cos": [], "ent": [], "ana": []}
        for s in cands:
            scores["cos"].append((cspec.get(s, 0.0), s))
            scores["ent"].append((espec.get(s, 0.0), s))
        # analog propagation over train + coconut candidates
        for s in cands + cc:
            f = tfp.get(s, cfp.get(s))
            best = 0.0
            if f is not None:
                for sim, a in analogs:
                    g = tfp.get(a)
                    if g is None: continue
                    v = (sim ** p_pow) * tanimoto(f, g)
                    if v > best: best = v
            scores["ana"].append((best, s))
        for ch in scores:
            scores[ch].sort(reverse=True)
            rank = next((i + 1 for i, (_, s) in enumerate(scores[ch]) if s == qs), 10**9)
            rr, hh = res[ch][0], res[ch][1]
            rr.append(1 / rank if rank <= 25 else 0.0)
            hh.append(rank)
        if (qi + 1) % 10 == 0:
            print(f"{qi+1}/{n_query} " + " ".join(f"{c}={np.mean(res[c][0]):.3f}" for c in res), flush=True)
    for ch in res:
        rr = np.array(res[ch][0]); hh = np.array(res[ch][1])
        print(f"{ch}: MRR25={rr.mean():.3f} hit@1={(hh<=1).mean():.3f} hit@5={(hh<=5).mean():.3f} hit@25={(hh<=25).mean():.3f}")


if __name__ == "__main__":
    run()
