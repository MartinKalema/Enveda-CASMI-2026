"""v8 fingerprint transformer: softmax-CE over hard in-window decoys.
Upgrades over v2: (1) decoy loss not BCE, (2) augmentation, (3) best
checkpoint on retrieval hit@5, (4) dot-product ranking with raw logits.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from bisect import bisect_left, bisect_right
from v2.train_fp import bin_spectrum, meta_vec, N_BINS, ADDUCTS, load_frame, FpMLP
from v1.subformula import ADDUCT_DELTA
from v7.floor import formula_exact_mass as _fem

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
N_DECOY = 63


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def featurize(df, augment=False, rng=None):
    X = np.zeros((len(df), N_BINS + len(ADDUCTS) + 2), dtype=np.float32)
    for i, r in enumerate(df.itertuples()):
        mz = np.asarray(r.ms2_mzs, dtype=float); it = np.asarray(r.ms2_normalized_intensities, dtype=float)
        if augment:
            keep = rng.random(len(mz)) > 0.2
            mz, it = mz[keep], it[keep] * rng.uniform(0.8, 1.2, size=keep.sum())
            mz = mz * (1 + rng.normal(0, 5e-6, size=len(mz)))
        X[i, :N_BINS] = bin_spectrum(mz, it)
        X[i, N_BINS:] = meta_vec(r.adduct, r.precursor_mz, r.collision_energy_ev)
    return X


def build_pool(tr, fmap):
    from v7.floor import formula_exact_mass
    first = tr.groupby("normalized_smiles")["molecular_formula"].first()
    mass = {}
    for s, f in first.items():
        if s not in fmap:
            continue
        m = formula_exact_mass(f)
        if m is not None:
            mass[s] = m
    pairs = sorted(mass.items(), key=lambda kv: kv[1])
    return [p[1] for p in pairs], [p[0] for p in pairs], np.stack(
        [fmap[p[0]] for p in pairs])


def val_hits(model, device, va, pool_m, pool_s, pool_F, k=5, n=300):
    model.eval()
    q = va.sample(min(n, len(va)), random_state=0)
    X = featurize(q)
    hits = 0
    from v7.floor import formula_exact_mass as _fem
    with torch.no_grad():
        Z = model(torch.from_numpy(X).to(device)).cpu().numpy()
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    checked = 0
    for zi, (_, r) in zip(Zn, q.iterrows()):
        qm = _fem(r["molecular_formula"]) if "molecular_formula" in q.columns else None
        if qm is None:
            qm = neutral_mass(r["precursor_mz"], r["adduct"])
        tol = max(qm * 10 / 1e6, 0.01)
        lo = bisect_left(pool_m, qm - tol); hi = bisect_right(pool_m, qm + tol)
        idx = list(range(lo, min(hi, lo + 256)))
        if not idx: continue
        checked += 1
        F = pool_F[idx]
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        scores = Fn @ zi
        order = np.argsort(-scores)
        cand = [pool_s[i] for i in [idx[j] for j in order[:k]]]
        if r["normalized_smiles"] in cand: hits += 1
    print(f"  [val checked {checked}/{len(q)}]", flush=True)
    return hits / max(len(q), 1)


def main(epochs=15, n_train=400000, batch=128, seed=0):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device", device, flush=True)
    tr, fmap = load_frame()
    tr["neutral_q"] = [neutral_mass(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    groups = np.array(tr["inchikey14"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val_g = set(groups[:max(50, len(groups) // 20)])
    va = tr[tr["inchikey14"].isin(val_g)].reset_index(drop=True)
    pool = tr[~tr["inchikey14"].isin(val_g)].reset_index(drop=True)
    va_ret = pool.sample(3000, random_state=seed).reset_index(drop=True)  # truth IN pool
    trn = pool.sample(min(n_train, len(pool)), random_state=seed).reset_index(drop=True)
    pool_m, pool_s, pool_F = build_pool(pool, fmap)
    pool_F = pool_F.astype(np.float32)
    print(f"train {len(trn)} val {len(va)} val_ret {len(va_ret)} pool {len(pool_s)}", flush=True)
    model = FpMLP(d_h=1536).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_ep = -1, -1
    for ep in range(epochs):
        model.train()
        batch_idx = np.random.default_rng(seed + ep).choice(len(trn), size=(len(trn) // batch, batch))
        tot, nb = 0.0, 0
        for bi in batch_idx:
            df = trn.iloc[bi]
            X = featurize(df, augment=True, rng=np.random.default_rng())
            Xt = torch.from_numpy(X).to(device)
            Z = model(Xt)  # raw logits for dot ranking
            # decoys per query from pool mass window
            cand_idx, cand_pos = [], []
            for j, (_, r) in enumerate(df.iterrows()):
                qm = _fem(r["molecular_formula"]) if "molecular_formula" in df.columns else None
                if qm is None:
                    qm = r["neutral_q"]
                tol = max(qm * 10 / 1e6, 0.01)
                lo = bisect_left(pool_m, qm - tol); hi = bisect_right(pool_m, qm + tol)
                SEA = list(range(lo, hi))
                if len(SEA) > N_DECOY + 1:
                    SEA = list(np.random.default_rng().choice(SEA, size=N_DECOY + 1, replace=False))
                # ensure truth included
                cand_idx.append(SEA)
            maxc = max(len(c) for c in cand_idx)
            B = len(df)
            F = np.zeros((B, maxc, 2048), dtype=np.float32)
            tgt = np.zeros(B, dtype=np.int64)
            for j, SEA in enumerate(cand_idx):
                F[j, :len(SEA)] = pool_F[SEA]
                names = [pool_s[i] for i in SEA]
                t = df.iloc[j]["normalized_smiles"]
                tgt[j] = names.index(t) if t in names else 0
                if t not in names: F[j, 0] = fmap[t]
            # cosine + temperature: stops magnitude shortcut (popcount ranking)
            Zn = Z / (Z.norm(dim=1, keepdim=True) + 1e-9)
            Fn = torch.from_numpy(F).to(device)
            Fn = Fn / (Fn.norm(dim=2, keepdim=True) + 1e-9)
            logits = torch.einsum("bd,bcd->bc", Zn, Fn) / 0.1
            loss = nn.CrossEntropyLoss()(logits, torch.from_numpy(tgt).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        h5 = val_hits(model, device, va_ret, pool_m, pool_s, pool_F)
        print(f"ep{ep+1} loss={tot/nb:.4f} val_hit@5={h5:.3f}", flush=True)
        if h5 > best:
            best, best_ep = h5, ep
            torch.save(model.state_dict(), f"{PROJECT}/v8/fp_trans.pt")
    print(f"best val_hit@5={best:.3f} ep{best_ep+1}", flush=True)


if __name__ == "__main__":
    main()
