"""v-fpnet: fingerprint network done RIGHT.

Set-transformer encoder over subformula-annotated peaks + covariates,
trained with 63 same-mass hard decoys under softmax CE, augmentation,
best-checkpoint on retrieval hit@5 (pooled-truth AND disjoint),
ranking by raw-logit dot product f.z (finding 5).

Saves: v-fpnet/fpnet_best.pt, v-fpnet/meta.json, v-fpnet/subfrag_masses.npy
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from bisect import bisect_left, bisect_right

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
VDIR = f"{PROJECT}/v-fpnet"
N_DECOY = 63
TOPK_PEAKS = 80
PF_DIM = 8
D_MODEL = 256
TAU = 10.0  # temperature on raw f.z dot logits (train only; rank is raw f.z)

import sys
sys.path.insert(0, PROJECT)
from v1.subformula import ADDUCT_DELTA, enumerate_subformulae, parse_formula
from v7.floor import formula_exact_mass

ADDUCT_TOP = None  # set at runtime
INSTR_BUCKETS = ["timstof", "orbitrap", "qtof", "lc-esi-qtof", "esi-qft",
                 "lc-esi-qft", "lc-esi-itft", "tof"]  # + OTHER


# ---------------- subformula fragment-mass library (truth-free) ----------------
def build_subfrag_library(formulae, top_n=150, per_f=8000, path=None):
    from collections import Counter
    top = [f for f, _ in Counter(formulae).most_common(top_n)]
    masses = []
    for f in top:
        pf = parse_formula(f)
        if pf is None:
            continue
        try:
            _, subs = enumerate_subformulae(pf, max_n=per_f)
        except Exception:
            continue
        masses.extend(m for _, m in subs)
    arr = np.unique(np.asarray(masses, dtype=np.float64))
    arr = arr[(arr > 10) & (arr < 2000)]
    arr = arr.astype(np.float32)
    if path:
        np.save(path, arr)
    print(f"subfrag library: {len(arr)} masses from {len(top)} formulae", flush=True)
    return arr


def annotate_peaks(mzs, intens, adduct, prec_mz, lib):
    """Per-peak subformula annotation WITHOUT truth formula.

    Neutralises each peak by the precursor adduct delta and matches against a
    precomputed library of plausible subformula masses (union over frequent
    training formulae). Returns (hit, ppm) per peak.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    n = len(mzs)
    hit = np.zeros(n, dtype=np.float32)
    lppm = np.zeros(n, dtype=np.float32)
    d = ADDUCT_DELTA.get(adduct)
    if d is None or len(lib) == 0 or n == 0:
        defect = (mzs - np.floor(np.maximum(mzs, 0))).astype(np.float32)
        return hit, lppm, defect
    neutral = mzs - d
    defect = (neutral - np.floor(np.maximum(neutral, 0))).astype(np.float32)
    pos = np.searchsorted(lib, neutral)
    j0 = np.clip(pos - 1, 0, len(lib) - 1)
    j1 = np.clip(pos, 0, len(lib) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e0 = np.abs(lib[j0].astype(np.float64) - neutral) / np.maximum(neutral, 1e-9) * 1e6
        e1 = np.abs(lib[j1].astype(np.float64) - neutral) / np.maximum(neutral, 1e-9) * 1e6
    best = np.minimum(e0, e1)
    ok = (best <= 15.0) & (neutral > 0)
    hit[ok] = 1.0
    lppm[ok] = np.log1p(best[ok] / 15.0).astype(np.float32)
    return hit, lppm, defect


# ---------------- featurization ----------------
def ce_stats(ce):
    try:
        if ce is None:
            return 0.0, 0, 0.0
        a = np.atleast_1d(np.asarray(ce, dtype=float)).ravel()
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return 0.0, 0, 0.0
        return float(a.mean()), len(a), 1.0
    except Exception:
        return 0.0, 0, 0.0


def featurize_block(df, lib, adduct_top):
    n = len(df)
    PF = np.zeros((n, TOPK_PEAKS, PF_DIM), dtype=np.float32)
    PM = np.zeros((n, TOPK_PEAKS), dtype=bool)
    COV = np.zeros((n, 16 + 9 + 6), dtype=np.float32)
    for i, r in enumerate(df.itertuples()):
        mz = np.asarray(r.ms2_mzs, dtype=float)
        it = np.asarray(r.ms2_normalized_intensities, dtype=float)
        ok = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (it > 0)
        mz, it = mz[ok], it[ok]
        if len(mz) > TOPK_PEAKS:
            keep = np.argpartition(-it, TOPK_PEAKS)[:TOPK_PEAKS]
            mz, it = mz[keep], it[keep]
        o = np.argsort(-it)  # strongest first (dropout keeps first 4)
        mz, it = mz[o], it[o]
        m = len(mz)
        hit, lppm, defect = annotate_peaks(mz, it, r.adduct, r.precursor_mz, lib)
        mx = it.max() if m else 1.0
        inten = it / mx
        nl = np.clip((r.precursor_mz - mz) / 1200.0, 0, 2)
        PF[i, :m, 0] = mz / 1200.0
        PF[i, :m, 1] = inten
        PF[i, :m, 2] = np.sqrt(np.maximum(inten, 0))
        PF[i, :m, 3] = nl
        PF[i, :m, 4] = defect
        PF[i, :m, 5] = hit
        PF[i, :m, 6] = lppm
        PF[i, :m, 7] = np.arange(m) / TOPK_PEAKS
        PM[i, :m] = True
        a = adduct_top.index(r.adduct) if r.adduct in adduct_top else 15
        COV[i, a] = 1.0
        ins = str(r.instrument_type).lower().strip() if r.instrument_type else "unk"
        b = INSTR_BUCKETS.index(ins) if ins in INSTR_BUCKETS else 8
        COV[i, 16 + b] = 1.0
        COV[i, 25] = 1.0 if str(getattr(r, "ionization_mode", "")).lower().startswith("pos") else 0.0
        COV[i, 26] = r.precursor_mz / 1000.0
        cmean, cn, has = ce_stats(r.collision_energy_ev)
        COV[i, 27] = cmean / 100.0
        COV[i, 28] = min(cn, 4) / 4.0
        COV[i, 29] = has
        COV[i, 30] = min(len(mz), 200) / 200.0
    return PF, PM, COV


# ---------------- model ----------------
class EncLayer(nn.Module):
    """Pre-norm transformer block with additive pad mask (MPS-safe)."""

    def __init__(self, d, heads, p=0.1):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
                                nn.Dropout(p), nn.Linear(512, d))
        self.do = nn.Dropout(p)

    def forward(self, x, pad):
        # pad: (B, L) bool, True = real peak
        B, L, _ = x.shape
        neg = torch.zeros(B, L, device=x.device, dtype=x.dtype)
        neg = neg.masked_fill(~pad, float("-inf"))
        amask = neg[:, None, :].repeat(1, L, 1).repeat_interleave(self.h, dim=0)
        a, _ = self.att(self.ln1(x), self.ln1(x), self.ln1(x),
                        attn_mask=amask)
        x = x + self.do(a)
        x = x + self.do(self.ff(self.ln2(x)))
        return x


class FpSetNet(nn.Module):
    def __init__(self, d=D_MODEL, layers=2, heads=4, cov_dim=31, out=2048, p=0.1):
        super().__init__()
        self.h = heads
        self.emb = nn.Linear(PF_DIM, d)
        self.tr = nn.ModuleList([EncLayer(d, heads, p) for _ in range(layers)])
        self.query = nn.Parameter(torch.randn(1, 1, d) / d ** 0.5)
        self.pma = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.cov = nn.Sequential(nn.Linear(cov_dim, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d + 64, 1024), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(1024, out))

    def forward(self, pf, mask, cov):
        h = self.emb(pf)
        for enc in self.tr:
            h = enc(h, mask)
        B, L = mask.shape
        neg = torch.zeros(B, L, device=h.device, dtype=h.dtype)
        neg = neg.masked_fill(~mask, float("-inf"))
        amask = neg[:, None, :].repeat_interleave(self.h, dim=0)
        q = self.query.expand(pf.size(0), -1, -1)
        s, _ = self.pma(q, h, h, attn_mask=amask)
        s = s[:, 0]
        c = self.cov(cov)
        return self.head(torch.cat([s, c], dim=1))


# ---------------- splits / pool ----------------
def neutral_mass_row(prec, adduct, formula):
    m = formula_exact_mass(formula) if isinstance(formula, str) else None
    if m is not None:
        return m
    if adduct == "[2M+H]+":
        return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+":
        return (prec - 22.989218) / 2
    if adduct == "[2M-H]-":
        return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def make_splits(seed=0):
    cols = ["normalized_smiles", "inchikey14", "molecular_formula", "adduct",
            "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
            "collision_energy_ev", "instrument_type", "ionization_mode"]
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet", columns=cols)
    fp = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    fmap = {s: np.packbits(np.unpackbits(np.asarray(f, dtype=np.uint8)).astype(np.uint8))
            for s, f in zip(fp["smiles"], fp["fp"])}
    tr = tr[tr["normalized_smiles"].isin(fmap)].reset_index(drop=True)
    tr["neutral_q"] = [neutral_mass_row(p, a, f) for p, a, f in
                       zip(tr["precursor_mz"], tr["adduct"], tr["molecular_formula"])]
    tr = tr[np.isfinite(tr["neutral_q"])].reset_index(drop=True)
    groups = np.array(tr["inchikey14"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val_g = set(groups[:max(50, len(groups) // 20)])
    is_val = tr["inchikey14"].isin(val_g)
    pool = tr[~is_val].reset_index(drop=True)
    vdis = tr[is_val].reset_index(drop=True)
    return tr, pool, vdis, fmap


def build_pool(pool, fmap):
    grp = pool.groupby("normalized_smiles")
    first = grp["molecular_formula"].first()
    medq = grp["neutral_q"].median()
    smiles, masses, packs = [], [], []
    for s, f in first.items():
        if s not in fmap:
            continue
        m = formula_exact_mass(f) if isinstance(f, str) else None
        if m is None:
            m = float(medq[s])  # fallback: median adduct-derived neutral mass
        if not np.isfinite(m):
            continue
        smiles.append(s)
        masses.append(m)
        packs.append(fmap[s])
    o = np.argsort(masses)
    return (np.asarray(masses)[o], [smiles[i] for i in o],
            np.stack([packs[i] for i in o]))


def sample_decoys(qmass, pool_m, pool_s, truth, rng):
    tol = max(qmass * 10 / 1e6, 0.01)
    lo = bisect_left(pool_m, qmass - tol)
    hi = bisect_right(pool_m, qmass + tol)
    sea = list(range(lo, hi))
    others = [i for i in sea if pool_s[i] != truth]
    if len(others) > N_DECOY:
        others = list(rng.choice(others, size=N_DECOY, replace=False))
    return others  # truth injected at pos 0 by caller


# ---------------- retrieval eval (raw f.z) ----------------
def unpack_mat(packed):
    return np.unpackbits(np.ascontiguousarray(packed), axis=-1).astype(np.float32)


@torch.no_grad()
def predict(model, PF, PM, COV, device, batch=256):
    model.eval()
    out = []
    for i in range(0, len(PF), batch):
        sl = slice(i, i + batch)
        z = model(torch.from_numpy(PF[sl]).to(device),
                  torch.from_numpy(PM[sl]).to(device),
                  torch.from_numpy(COV[sl]).to(device))
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def hits_at_k(Z, queries, pool_m, pool_s, pool_P, fmap, ks=(1, 5, 10), inject_truth=True):
    Zn = Z  # raw logits; rank by f.z (f = 0/1 bits)
    res = {k: 0 for k in ks}
    n = 0
    for zi, (_, r) in zip(Zn, queries.iterrows()):
        qm = r["neutral_q"]
        tol = max(qm * 10 / 1e6, 0.01)
        lo = bisect_left(pool_m, qm - tol)
        hi = bisect_right(pool_m, qm + tol)
        idx = list(range(lo, hi))
        if inject_truth:
            idx = [i for i in idx if pool_s[i] != r["normalized_smiles"]]
            F = unpack_mat(pool_P[idx]) if idx else np.zeros((0, 2048), np.float32)
            F = np.concatenate([fmap[r["normalized_smiles"]].astype(np.float32)
                                [None, :], F])
            names = [r["normalized_smiles"]] + [pool_s[i] for i in idx]
        else:
            if not idx:
                continue
            F = unpack_mat(pool_P[idx])
            names = [pool_s[i] for i in idx]
        if len(names) > 5000:
            keep = np.random.default_rng(0).choice(len(names), 5000, replace=False)
            keep = np.sort(keep)
            F, names = F[keep], [names[i] for i in keep]
        scores = F @ zi  # raw-logit dot product
        order = np.argsort(-scores)
        n += 1
        for k in ks:
            if r["normalized_smiles"] in [names[j] for j in order[:k]]:
                res[k] += 1
    return {k: res[k] / max(n, 1) for k in ks}, n


# ---------------- training ----------------
def augment(PF, PM, rng):
    B = len(PF)
    PF = PF.copy()
    keep = rng.random(PF.shape[:2]) > 0.2
    keep[:, :4] = True  # always keep top-4 peaks
    keep &= PM
    u = rng.uniform(0.8, 1.2, size=(B, TOPK_PEAKS, 1)).astype(np.float32)
    PF[:, :, 1:2] *= np.where(keep[..., None], u, 1.0)
    PF[:, :, 2:3] *= np.where(keep[..., None], np.sqrt(u), 1.0)
    nz = (1 + rng.normal(0, 5e-6, size=(B, TOPK_PEAKS, 1))).astype(np.float32)
    PF[:, :, 0:1] *= np.where(keep[..., None], nz, 1.0)
    return PF, keep


def main(epochs=16, n_train=150000, batch=128, seed=0, resume=False, init_from=""):
    import os
    os.makedirs(VDIR, exist_ok=True)
    import os
    os.makedirs(VDIR, exist_ok=True)
    import os
    os.makedirs(VDIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device", device, flush=True)
    tr, pool, vdis, fmap = make_splits(seed)
    global ADDUCT_TOP
    ADDUCT_TOP = list(pool["adduct"].value_counts().head(15).index)
    print(f"pool {len(pool)} vdis {len(vdis)} structures pool=", flush=True)
    pool_m, pool_s, pool_P = build_pool(pool, fmap)
    print(f"pool_structures {len(pool_s)}", flush=True)
    smi2idx = {s: i for i, s in enumerate(pool_s)}

    lib_path = f"{VDIR}/subfrag_masses.npy"
    try:
        lib = np.load(lib_path)
        print(f"subfrag library loaded: {len(lib)}", flush=True)
    except Exception:
        lib = build_subfrag_library(pool["molecular_formula"].tolist(), path=lib_path)

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(pool), size=min(n_train, len(pool)), replace=False)
    train_idx.sort()
    trn = pool.iloc[train_idx].reset_index(drop=True)
    rest = pool.drop(pool.index[train_idx])
    q_pooled = rest.sample(800, random_state=seed)
    q_dis = vdis.sample(800, random_state=seed)
    print(f"train {len(trn)} q_pooled {len(q_pooled)} q_dis {len(q_dis)}", flush=True)

    print("featurizing train...", flush=True)
    PFtr, PMtr, COVtr = featurize_block(trn, lib, ADDUCT_TOP)
    print("featurizing queries...", flush=True)
    PFp, PMp, COVp = featurize_block(q_pooled, lib, ADDUCT_TOP)
    PFd, PMd, COVd = featurize_block(q_dis, lib, ADDUCT_TOP)
    # truth bits for disjoint injection (unpacked float)
    fmap_f = {s: np.unpackbits(p).astype(np.float32) for s, p in fmap.items()}

    model = FpSetNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    if init_from:
        model.load_state_dict(torch.load(init_from, map_location=device))
        print(f"warm start from {init_from}", flush=True)
    celoss = nn.CrossEntropyLoss()
    PFt = torch.from_numpy(PFtr)
    PMt = torch.from_numpy(PMtr)
    COVt = torch.from_numpy(COVtr)
    qmass = trn["neutral_q"].to_numpy()
    truth_idx = np.array([smi2idx[s] for s in trn["normalized_smiles"]])
    n = len(trn)
    steps = n // batch

    best_p, best_ep, best_d = -1, -1, None
    start_ep = 0
    if init_from and not resume:
        try:
            with open(f"{VDIR}/meta.json") as f:
                mb = json.load(f)
            best_p = float(mb["pooled"]["5"])
            best_ep = int(mb["ep"]) - 1
            best_d = {int(k): v for k, v in mb["disjoint"].items()}
            print(f"carry best pooled@5={best_p:.3f} ep{best_ep+1}", flush=True)
        except Exception as e:
            print(f"no prior best ({e})", flush=True)
    if resume:
        try:
            ck = torch.load(f"{VDIR}/fpnet_last.pt", map_location=device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            sched.T_max = epochs
            start_ep = ck["ep"]
            best_p, best_ep, best_d = ck["best_p"], ck["best_ep"], ck["best_d"]
            print(f"resumed at ep{start_ep+1} best_pooled@5={best_p:.3f} ep{best_ep+1}",
                  flush=True)
        except Exception as e:
            print(f"resume failed ({e}), starting fresh", flush=True)
    for ep in range(start_ep, epochs):
        # fresh hard-decoy sample each epoch
        erng = np.random.default_rng(seed * 1000 + ep)
        CIDX = np.zeros((n, N_DECOY + 1), dtype=np.int64)
        CIDX[:, 0] = truth_idx
        for j in range(n):
            if j % 50000 == 0:
                print(f"  decoy {j}/{n}", flush=True)
            CIDX[j, 1:] = 0
            oth = sample_decoys(qmass[j], pool_m, pool_s,
                                trn["normalized_smiles"].iloc[j], erng)
            CIDX[j, 1:1 + len(oth)] = [i for i in oth]
            # pad remainder with repeats of drawn decoys (or truth)
            for k in range(1 + len(oth), N_DECOY + 1):
                CIDX[j, k] = CIDX[j, k - 1] if len(oth) else truth_idx[j]
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        arng = np.random.default_rng(seed + ep)
        for s in range(steps):
            bi = perm[s * batch:(s + 1) * batch].numpy()
            pf_a, pm_a = augment(PFtr[bi], PMtr[bi], arng)
            Z = model(torch.from_numpy(pf_a).to(device),
                      torch.from_numpy(pm_a).to(device),
                      torch.from_numpy(COVtr[bi]).to(device))
            Fb = unpack_mat(pool_P[CIDX[bi]])
            Fbt = torch.from_numpy(Fb).to(device)
            logits = torch.einsum("bd,bcd->bc", Z, Fbt) / TAU
            tgt = torch.zeros(len(bi), dtype=torch.long, device=device)
            loss = celoss(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        Zp = predict(model, PFp, PMp, COVp, device)
        Zd = predict(model, PFd, PMd, COVd, device)
        hp, _ = hits_at_k(Zp, q_pooled, pool_m, pool_s, pool_P, fmap_f,
                           inject_truth=False)
        hd, _ = hits_at_k(Zd, q_dis, pool_m, pool_s, pool_P, fmap_f,
                           inject_truth=True)
        print(f"ep{ep+1} loss={tot/steps:.4f} pooled@1/5/10="
              f"{hp[1]:.3f}/{hp[5]:.3f}/{hp[10]:.3f} "
              f"disjoint@1/5/10={hd[1]:.3f}/{hd[5]:.3f}/{hd[10]:.3f}", flush=True)
        if hp[5] > best_p:
            best_p, best_ep, best_d = hp[5], ep, hd
            torch.save(model.state_dict(), f"{VDIR}/fpnet_best.pt")
            with open(f"{VDIR}/meta.json", "w") as f:
                json.dump({"adduct_top": ADDUCT_TOP, "instr": INSTR_BUCKETS,
                           "seed": seed, "ep": ep + 1,
                           "pooled": hp, "disjoint": hd}, f, indent=1)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "ep": ep + 1,
                    "best_p": best_p, "best_ep": best_ep, "best_d": best_d},
                   f"{VDIR}/fpnet_last.pt")
    print(f"BEST pooled@5={best_p:.3f} ep{best_ep+1} disjoint@1/5/10="
          f"{best_d[1]:.3f}/{best_d[5]:.3f}/{best_d[10]:.3f}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", type=str, default="")
    a = ap.parse_args()
    main(epochs=a.epochs, resume=a.resume, init_from=a.init)
