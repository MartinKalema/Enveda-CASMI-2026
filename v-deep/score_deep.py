"""MS2DeepScore vs cosine, frozen disjoint queries (n=30, seed=30 to match metfrag run)."""
import numpy as np, pandas as pd, torch
from bisect import bisect_left, bisect_right
from matchms import Spectrum
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def ms2s(mz, it, prec, adduct=""):
    ion = "negative" if adduct.rstrip().endswith("]-") else "positive"
    return Spectrum(mz=np.asarray(mz, dtype=float), intensities=np.asarray(it, dtype=float),
                    metadata={"precursor_mz": float(prec), "ionmode": ion})


def run(n_query=30, seed=30):
    model = load_model(f"{PROJECT}/data/ms2deepscore_model.pt", allow_legacy=True)
    scorer = MS2DeepScore(model)
    train = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities"])
    train["neutral"] = [neutral_mass(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    train = train[np.isfinite(train["neutral"].values)]
    rng = np.random.default_rng(seed)
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

    mrr_d, mrr_c, corrs = [], [], []
    for qi, qs in enumerate(queries):
        qspec = qpool[qpool["normalized_smiles"] == qs]
        q = qspec.iloc[0]
        qmass = float(np.median([neutral_mass(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        tol = qmass * 20 / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        cands = list(smi[lo:hi][:300])
        win = db_samp[db_samp["normalized_smiles"].isin(set(cands))]
        # deep scores via pair() (binned SpectrumDocument under the hood)
        qobj = ms2s(q["ms2_mzs"], q["ms2_normalized_intensities"], q["precursor_mz"], q["adduct"])
        best = {}
        for s in cands:
            b = 0.0
            for _, t in win[win["normalized_smiles"] == s].iterrows():
                v = float(scorer.pair(qobj, ms2s(t["ms2_mzs"], t["ms2_normalized_intensities"],
                                                 t["precursor_mz"], t["adduct"])))
                if v > b:
                    b = v
            best[s] = b
        # cosine baseline
        csco = {}
        for s in cands:
            b = 0.0
            for _, t in win[win["normalized_smiles"] == s].iterrows():
                c = cosine(q["ms2_mzs"], q["ms2_normalized_intensities"],
                           t["ms2_mzs"], t["ms2_normalized_intensities"])
                if c > b:
                    b = c
            csco[s] = b
        rd = sorted(((v, s) for s, v in best.items()), reverse=True)
        rc = sorted(((v, s) for s, v in csco.items()), reverse=True)
        rk_d = next((i + 1 for i, (_, s) in enumerate(rd) if s == qs), 10**9)
        rk_c = next((i + 1 for i, (_, s) in enumerate(rc) if s == qs), 10**9)
        mrr_d.append(1 / rk_d if rk_d <= 25 else 0.0)
        mrr_c.append(1 / rk_c if rk_c <= 25 else 0.0)
        x = np.array([best.get(s, 0) for s in cands]); y = np.array([csco.get(s, 0) for s in cands])
        if x.std() > 0 and y.std() > 0:
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
        if (qi + 1) % 10 == 0:
            print(f"{qi+1}/{n_query} deep={np.mean(mrr_d):.3f} cos={np.mean(mrr_c):.3f}", flush=True)
    print(f"ms2deepscore MRR={np.mean(mrr_d):.3f} cosine MRR={np.mean(mrr_c):.3f} corr={np.mean(corrs):.3f}")


if __name__ == "__main__":
    run()
