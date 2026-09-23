"""Learned fingerprint predictor (MIST path): binned spectrum -> 2048-bit Morgan fp.
Torch MLP, MPS/CPU, structure-disjoint split. Saves v2/fp_mlp.pt
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, os

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
BIN_W, MZ_MAX = 0.1, 1200.0
N_BINS = int(MZ_MAX / BIN_W)


def bin_spectrum(mzs, intens, out=None):
    v = np.zeros(N_BINS, dtype=np.float32) if out is None else out
    if out is not None: v[:] = 0
    m = np.asarray(mzs, dtype=float); it = np.asarray(intens, dtype=float)
    idx = (m / BIN_W).astype(int)
    ok = (idx >= 0) & (idx < N_BINS)
    np.add.at(v, idx[ok], it[ok])
    n = float(np.sqrt((v * v).sum()))
    if n > 0: v /= n
    return v


ADDUCTS = ["[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M-H]-", "[M+Cl]-",
           "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M]+", "[M-H2O+H]+"]


def meta_vec(adduct, precursor_mz, ce_list):
    a = [1.0 if adduct == x else 0.0 for x in ADDUCTS]
    try: ce = float(np.mean(list(ce_list))) if ce_list is not None else 0.0
    except Exception: ce = 0.0
    return np.array(a + [precursor_mz / 1000.0, ce / 100.0], dtype=np.float32)


class FpMLP(nn.Module):
    def __init__(self, d_in=N_BINS + len(ADDUCTS) + 2, d_h=1024, d_out=2048, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h, d_h // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h // 2, d_out))
    def forward(self, x): return self.net(x)


def load_frame(n_cap=None, seed=0):
    cols = ["normalized_smiles", "inchikey14", "molecular_formula", "adduct", "precursor_mz",
            "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet", columns=cols)
    fp = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    fmap = {s: np.unpackbits(np.asarray(f, dtype=np.uint8)).astype(np.float32)
            for s, f in zip(fp["smiles"], fp["fp"])}
    tr = tr[tr["normalized_smiles"].isin(fmap)]
    if n_cap: tr = tr.sample(n_cap, random_state=seed).reset_index(drop=True)
    return tr, fmap


def featurize(df):
    X = np.zeros((len(df), N_BINS + len(ADDUCTS) + 2), dtype=np.float32)
    for i, r in enumerate(df.itertuples()):
        X[i, :N_BINS] = bin_spectrum(r.ms2_mzs, r.ms2_normalized_intensities)
        X[i, N_BINS:] = meta_vec(r.adduct, r.precursor_mz, r.collision_energy_ev)
    return X


def main(epochs=5, n_train=200000, batch=512, seed=0):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device", device, flush=True)
    tr, fmap = load_frame()
    groups = np.array(tr["inchikey14"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val_g = max(50, len(groups) // 20)
    val_g = set(groups[:n_val_g])
    va = tr[tr["inchikey14"].isin(val_g)].sample(3000, random_state=seed)
    pool = tr[~tr["inchikey14"].isin(val_g)]
    trn = pool.sample(min(n_train, len(pool)), random_state=seed)
    print(f"train {len(trn)} val {len(va)}", flush=True)
    Xtr, Ytr = featurize(trn), np.stack([fmap[s] for s in trn["normalized_smiles"]])
    Xva, Yva = featurize(va), np.stack([fmap[s] for s in va["normalized_smiles"]])
    model = FpMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xtr); Yt = torch.from_numpy(Ytr)
    Xv = torch.from_numpy(Xva); Yv = torch.from_numpy(Yva)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx].to(device)), Yt[idx].to(device))
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv.to(device))).cpu().numpy()
        tan = (np.logical_and(pv > 0.5, Yva > 0.5).sum(1) /
               np.maximum(np.logical_or(pv > 0.5, Yva > 0.5).sum(1), 1)).mean()
        print(f"ep{ep+1} loss={tot/n:.4f} val_tanimoto={tan:.3f}", flush=True)
    torch.save(model.state_dict(), f"{PROJECT}/v2/fp_mlp.pt")
    print("saved v2/fp_mlp.pt", flush=True)


if __name__ == "__main__":
    main()
