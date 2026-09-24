"""Head-to-head: cos/ent/ana/exp/metfrag/fpdot on identical frozen queries.
3 seeds x 40 queries, same candidates. One table to rule the channels.
"""
import numpy as np, pandas as pd, pickle, torch
from v1.subformula import explained_intensity
from v2.blend import cosine, tanimoto
from v4.channels import entropy_similarity
from v7.submit_v7 import frag_match
from v2.train_fp import FpMLP, bin_spectrum, meta_vec, N_BINS, ADDUCTS

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)


def run():
    device = "cpu"
    model = FpMLP(d_h=1536).to(device)
    model.load_state_dict(torch.load(f"{PROJECT}/data/fp_trans.pt", map_location=device))
    model.eval()
    with open(f"{PROJECT}/data/frag_cache.pkl", "rb") as f:
        frags = pickle.load(f)
    tf = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    scol = "normalized_smiles" if "normalized_smiles" in tf.columns else "smiles"
    tfp = {s: np.unpackbits(np.asarray(x, dtype=np.uint8)).astype(np.float32)
           for s, x in zip(tf[scol], tf["fp"])}
    out = {c: [] for c in ("cos", "ent", "ana", "exp", "metfrag", "fpdot")}
    for seed in SEEDS:
        d = pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{seed}.parquet")
        tr = pd.read_parquet(f"{PROJECT}/data/train.parquet",
            columns=["normalized_smiles", "adduct", "precursor_mz", "ms2_mzs",
                     "ms2_normalized_intensities", "collision_energy_ev"])
        tr = tr.set_index("normalized_smiles", append=False)
        fmap = {s: f for s, f in
                pd.read_parquet(f"{PROJECT}/data/train.parquet",
                                columns=["normalized_smiles", "molecular_formula"]
                                ).drop_duplicates("normalized_smiles").itertuples(index=False)}
        for (sd, truth), g in d.groupby(["seed", "truth"]):
            try:
                rows = tr.loc[[truth]].head(2)
            except KeyError:
                continue
            if len(rows) == 0:
                continue
            r = rows.iloc[0]
            with torch.no_grad():
                Z = model(torch.from_numpy(np.stack([_feat(r) for _, r in rows.iterrows()]))).numpy().mean(axis=0)
            Zn = Z / (np.linalg.norm(Z) + 1e-9)
            sc = {c: {} for c in out}
            for _, c in g.iterrows():
                s = c["cand"]
                sc["cos"][s] = c["cos"]; sc["ent"][s] = c["ent"]; sc["ana"][s] = c["ana"]
                sc["exp"][s] = explained_intensity(
                    r["ms2_mzs"], r["ms2_normalized_intensities"],
                    fmap.get(s, ""), r["adduct"])[0] if fmap.get(s) else 0.0
                sc["metfrag"][s] = frag_match(
                    r["ms2_mzs"], r["ms2_normalized_intensities"],
                    frags.get(s, []), r["adduct"])
                f = tfp.get(s)
                sc["fpdot"][s] = float((f / (np.linalg.norm(f) + 1e-9)) @ Zn) if f is not None else 0.0
            for ch in out:
                rnk = sorted(sc[ch].items(), key=lambda kv: -kv[1])
                rank = next((i + 1 for i, (s, _) in enumerate(rnk) if s == truth), 10**9)
                out[ch].append(1 / rank if rank <= 25 else 0.0)
        print(f"seed {seed} done", flush=True)
    print("CHANNEL MRR (150 queries):", flush=True)
    for ch, v in out.items():
        print(f"  {ch:8s} {np.mean(v):.3f}", flush=True)


def _feat(r):
    v = np.zeros(N_BINS + len(ADDUCTS) + 2, dtype=np.float32)
    v[:N_BINS] = bin_spectrum(r["ms2_mzs"], r["ms2_normalized_intensities"])
    v[N_BINS:] = meta_vec(r["adduct"], r["precursor_mz"], r["collision_energy_ev"])
    return v


if __name__ == "__main__":
    run()
