"""Head-to-head: their FPNet vs our MLP, identical frozen queries."""
import numpy as np
import pandas as pd
import torch
from bisect import bisect_left, bisect_right

from v1.subformula import ADDUCT_DELTA
from v2.train_fp import FpMLP, bin_spectrum, meta_vec, N_BINS, ADDUCTS
import v13.fork_fpnet_raw as FF

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


class Q:
    def __init__(self, r):
        self.ms2_mzs = r["ms2_mzs"]
        self.ms2_normalized_intensities = r["ms2_normalized_intensities"]
        self.precursor_mz = r["precursor_mz"]
        self.adduct = r["adduct"]
        self.instrument_type = r.get("instrument_type", "timsTOF")
        self.collision_energy_ev = r.get("collision_energy_ev", [25.0])
        self.ionization_mode = r.get("ionization_mode", "positive")


def ce_fallback(ce):
    try:
        return float(np.mean(np.atleast_1d(ce))) if ce is not None and len(np.atleast_1d(ce)) else 25.0
    except Exception:
        return 25.0


def run(n_query=30, seed=30):
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    FF.MACCSkeys = MACCSkeys
    FF.ExactMolWt = ExactMolWt
    FF.BITS = np.load("/tmp/cocofp/fp_bits.npy")
    FF._fp_init()
    device = "cpu"
    ck = torch.load("/tmp/fpmodels/fp_single_s2.pt", map_location="cpu", weights_only=False)
    tnet = FF.FPNet(ck["nbits"], d=ck["d"], layers=ck["layers"]).eval()
    tnet.load_state_dict(ck["model"])
    FF._MODEL = ([tnet], [], "cpu", ck["nbits"])
    mynet = FpMLP(d_h=1536).to(device)
    mynet.load_state_dict(torch.load(f"{PROJECT}/data/fp_trans.pt", map_location=device))
    mynet.eval()
    tf = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    scol = "normalized_smiles" if "normalized_smiles" in tf.columns else "smiles"
    myfp = {s: np.unpackbits(np.asarray(x, dtype=np.uint8)).astype(np.float32)
            for s, x in zip(tf[scol], tf["fp"])}
    rng = np.random.default_rng(seed)
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet")
    tr["neutral"] = [nm(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    tr = tr[np.isfinite(tr["neutral"].values)]
    groups = np.array(tr["inchikey14"].unique())
    held = set(rng.choice(groups, size=500, replace=False))
    qp = tr[tr["inchikey14"].isin(held)]
    structs = np.array(qp["normalized_smiles"].unique())
    rng.shuffle(structs)
    queries = list(structs[:n_query])
    db = tr[~tr["inchikey14"].isin(held)]
    struct = db.groupby("normalized_smiles")["neutral"].median().reset_index()
    truth = qp.groupby("normalized_smiles")["neutral"].median().reset_index()
    struct = pd.concat([struct, truth]).drop_duplicates("normalized_smiles")
    struct = struct.sort_values("neutral").reset_index(drop=True)
    masses = struct["neutral"].values
    smi = struct["normalized_smiles"].values
    aT, aM = [], []
    for qi, qs in enumerate(queries):
        qspec = qp[qp["normalized_smiles"] == qs]
        q = qspec.iloc[0]
        qmass = float(np.median([nm(p, a) for p, a in zip(qspec["precursor_mz"], qspec["adduct"])]))
        tol = qmass * 20 / 1e6
        lo = bisect_left(masses, qmass - tol)
        hi = bisect_right(masses, qmass + tol)
        cands = list(smi[lo:hi][:300])
        qo = Q(q)
        tz = FF._logits_raw([(q["ms2_mzs"], q["ms2_normalized_intensities"])],
                            [tnet], q["precursor_mz"], q["adduct"],
                            qo.instrument_type, ce_fallback(qo.collision_energy_ev), 1.0)
        v = np.zeros(N_BINS + len(ADDUCTS) + 2, dtype=np.float32)
        v[:N_BINS] = bin_spectrum(q["ms2_mzs"], q["ms2_normalized_intensities"])
        v[N_BINS:] = meta_vec(q["adduct"], q["precursor_mz"], q["collision_energy_ev"])
        with torch.no_grad():
            mz = mynet(torch.from_numpy(v[None, :])).numpy()[0]
        st, sm = [], []
        for s in cands:
            tfp, tm = FF.fp_and_mass(s)
            if tfp is not None and tz is not None:
                st.append((float(np.asarray(tfp, dtype=np.float32) @ tz), s))
            else:
                st.append((0.0, s))
            f = myfp.get(s)
            if f is not None:
                fn = f / (np.linalg.norm(f) + 1e-9)
                zn = mz / (np.linalg.norm(mz) + 1e-9)
                sm.append((float(fn @ zn), s))
            else:
                sm.append((0.0, s))
        st.sort(reverse=True)
        sm.sort(reverse=True)
        rt = next((i + 1 for i, (_, s) in enumerate(st) if s == qs), 10 ** 9)
        rm = next((i + 1 for i, (_, s) in enumerate(sm) if s == qs), 10 ** 9)
        aT.append(1 / rt if rt <= 25 else 0.0)
        aM.append(1 / rm if rm <= 25 else 0.0)
        if (qi + 1) % 10 == 0:
            print(f"{qi+1}/{n_query} theirs-fpnet={np.mean(aT):.3f} ours-mlp={np.mean(aM):.3f}", flush=True)
    print(f"THEIRS-FPNET MRR={np.mean(aT):.3f} OURS-MLP MRR={np.mean(aM):.3f}")


if __name__ == "__main__":
    run()
