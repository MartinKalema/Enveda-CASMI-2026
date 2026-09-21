"""Validate learned fingerprint retrieval (disjoint split, hit@k)."""
import numpy as np, pandas as pd, torch
from bisect import bisect_left, bisect_right
from v1.subformula import ADDUCT_DELTA
from v2.train_fp import FpMLP, bin_spectrum, meta_vec, N_BINS
from v2.blend import load_fp

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def tanimoto(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def run(n_query=30, seed=2):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = FpMLP().to(device)
    model.load_state_dict(torch.load(f"{PROJECT}/v2/fp_mlp.pt", map_location=device))
    model.eval()
    fp = load_fp()
    rng = np.random.default_rng(seed)
    train = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"])
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

    def predict(qspec):
        feats = []
        for _, r in qspec.iterrows():
            v = np.zeros(N_BINS + 12, dtype=np.float32)
            v[:N_BINS] = bin_spectrum(r["ms2_mzs"], r["ms2_normalized_intensities"])
            v[N_BINS:] = meta_vec(r["adduct"], r["precursor_mz"], r["collision_energy_ev"])
            feats.append(v)
        with torch.no_grad():
            out = torch.sigmoid(model(torch.from_numpy(np.stack(feats)).to(device))).cpu().numpy()
        return (out.mean(axis=0) > 0.4)

    hits = {1: 0, 5: 0, 25: 0}
    rr = []
    for qi, qs in enumerate(queries):
        qspec = qpool[qpool["normalized_smiles"] == qs]
        qmass = float(np.median([neutral_mass(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        tol = qmass * 20 / 1e6
        lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        pcur = 20
        while hi - lo < 200 and pcur < 500:
            pcur *= 2; tol = qmass * pcur / 1e6
            lo = bisect_left(masses, qmass - tol); hi = bisect_right(masses, qmass + tol)
        cands = list(smi[lo:hi][:2000])
        pred = predict(qspec)
        scored = sorted(((tanimoto(pred, fp[s]), s) for s in cands if s in fp), reverse=True)
        rank = next((i + 1 for i, (_, s) in enumerate(scored) if s == qs), 10**9)
        rr.append(1 / rank if rank <= 25 else 0.0)
        for k in hits:
            if rank <= k: hits[k] += 1
        if (qi + 1) % 10 == 0: print(f"{qi+1}/{n_query} MRR25={np.mean(rr):.3f}", flush=True)
    print(f"n={n_query} hit@1={hits[1]/n_query:.3f} hit@5={hits[5]/n_query:.3f} hit@25={hits[25]/n_query:.3f} MRR@25={np.mean(rr):.3f}")


if __name__ == "__main__":
    run()
