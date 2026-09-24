"""v-grid: staged grid search over upgrade combos with cached per-query channel scores.

Stage `cache`: grouped-by-molecule (inchikey14) splits, >=100 queries x 2 seeds.
  Per query caches (npz): candidate smiles + masses, cos, ent, analog x6 (Kxp),
  metfrag x3 (tol), fpdot, fptopk, mass features, truth, floor meta.
Stage `eval`: combine cached channels WITHOUT recompute -> ranked combo table,
  v0-cosine anchor on every split, ship iff delta>=+0.012 on >=2 seeds.
Stage `gbm`: GBM feature sets {3,8,25,31} trained leave-one-seed-out.

Usage: .venv/bin/python v-grid/run_grid.py cache|eval|gbm|report
"""
import os, sys, pickle, itertools
import numpy as np, pandas as pd
from bisect import bisect_left, bisect_right

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
sys.path.insert(0, PROJECT)
from v1.subformula import ADDUCT_DELTA
from v2.blend import cosine
from v4.channels import entropy_similarity
from v7.floor import neutral_from_precursor

SEEDS = (10, 11)
N_QUERY = 110
CACHE_DIR = f"{PROJECT}/v-grid/cache"
W_TOP = 100          # fills rerank top-W of floor base
CAND_CAP = 2000      # cache widest candidate set
MRR_CAP = 25
SHIP = 0.012

os.makedirs(CACHE_DIR, exist_ok=True)


def neutral(p, a):
    return neutral_from_precursor(p, a)


def pack_fp(bitarr):
    b = np.asarray(bitarr, dtype=np.uint8).reshape(-1)
    n = 2048
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))
    return np.packbits(b[:n]).view(np.uint64)


def tan_mat(C, A):
    """C: (n,32) uint64 cand fps, A: (m,32) -> (n,m) tanimoto float32."""
    inter = np.bitwise_and(C[:, None, :], A[None, :, :]).astype(np.uint64)
    union = np.bitwise_or(C[:, None, :], A[None, :, :]).astype(np.uint64)
    def popcnt(x):
        return np.unpackbits(x.view(np.uint8), axis=-1).sum(axis=-1)
    I = popcnt(inter).astype(np.float32)
    U = popcnt(union).astype(np.float32)
    return np.where(U > 0, I / np.maximum(U, 1), 0.0)


def load_static():
    print("load train meta...", flush=True)
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet", columns=[
        "normalized_smiles", "inchikey14", "molecular_formula", "adduct",
        "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
        "collision_energy_ev", "ionization_mode"])
    tr["neutral"] = [neutral(p, a) for p, a in zip(tr["precursor_mz"], tr["adduct"])]
    tr = tr[np.isfinite(tr["neutral"].values)].reset_index(drop=True)
    print(f"  train rows {len(tr)} structs {tr['normalized_smiles'].nunique()}", flush=True)
    print("load coconut + fps...", flush=True)
    cf = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet")
    tfp = pd.read_parquet(f"{PROJECT}/data/fingerprints.parquet")
    fp = {}
    for s, f in zip(tfp["smiles"], tfp["fp"]):
        fp[s] = np.unpackbits(np.asarray(f, dtype=np.uint8))[:2048]
    for s, f in zip(cf["canonical_smiles"], cf["fp"]):
        fp.setdefault(s, np.unpackbits(np.asarray(f, dtype=np.uint8))[:2048])
    print(f"  fp entries {len(fp)}", flush=True)
    with open(f"{PROJECT}/data/frag_cache.pkl", "rb") as fh:
        frags = pickle.load(fh)
    print(f"  frag entries {len(frags)}", flush=True)
    return tr, cf, fp, frags


def fp_model():
    import torch, torch.nn as nn
    sd = torch.load(f"{PROJECT}/data/fp_trans.pt", map_location="cpu")
    net = nn.Sequential(nn.Linear(12012, 1536), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(1536, 768), nn.ReLU(), nn.Dropout(0.2), nn.Linear(768, 2048))
    net.load_state_dict({k.replace("net.", "", 1): v for k, v in sd.items()})
    net.eval()
    return net


def featurize_spec(mzs, intens, adduct, prec, ce):
    from v2.train_fp import bin_spectrum, meta_vec
    x = np.zeros(12012, dtype=np.float32)
    x[:12000] = bin_spectrum(mzs, intens)
    try:
        x[12000:] = meta_vec(adduct, prec, ce)
    except Exception:
        pass
    return x


def cache_seed(seed, tr, cf, fp, frags, fpnet, coco_med, sform_all, cform_map):
    from v7.floor import formula_exact_mass as fem
    def sm(s):
        v = struct_med_db.get(s)
        if v is None or not np.isfinite(v):
            f = sform_all.get(s) or cform_map.get(s)
            try:
                v = fem(f) if f else None
            except Exception:
                v = None
        if v is None or not np.isfinite(v):
            v = coco_med.get(s, 0.0)
        return float(v)
    rng = np.random.default_rng(seed)
    groups = np.array(tr["inchikey14"].unique())
    rng.shuffle(groups)
    held = set(groups[:max(500, len(groups) // 20)])
    qpool = tr[tr["inchikey14"].isin(held)].reset_index(drop=True)
    db = tr[~tr["inchikey14"].isin(held)].reset_index(drop=True)
    structs = np.array(qpool["normalized_smiles"].unique())
    rng.shuffle(structs)
    queries = list(structs[:N_QUERY])
    print(f"seed {seed}: held {len(held)} queries {len(queries)}", flush=True)

    struct = db.groupby("normalized_smiles")["neutral"].median()
    struct_med_db = struct.to_dict()
    struct = struct.reset_index().sort_values("neutral").reset_index(drop=True)
    smass_all = struct["neutral"].values
    ssmi_all = struct["normalized_smiles"].values
    coco = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = coco["exact_molecular_weight"].values
    csmi = coco["canonical_smiles"].values
    cform = coco["molecular_formula"].values if "molecular_formula" in coco.columns else None
    db_samp = db.groupby("normalized_smiles").head(2).reset_index(drop=True)
    db_neu = db_samp["neutral"].values
    sik14 = dict(zip(db["normalized_smiles"], db["inchikey14"]))
    qik14 = dict(zip(qpool["normalized_smiles"], qpool["inchikey14"]))
    sik14.update(qik14)
    sform_tr = db.groupby("normalized_smiles")["molecular_formula"].first().to_dict()
    from collections import Counter
    fprior = Counter(db["molecular_formula"].tolist())
    cform_map = dict(zip(csmi, cform)) if cform is not None else {}

    out = []
    for qi, qs in enumerate(queries):
        qspec = qpool[qpool["normalized_smiles"] == qs].reset_index(drop=True)
        qmass = float(np.median([neutral(p, a) for p, a in
                                 zip(qspec["precursor_mz"], qspec["adduct"])]))
        qadd = qspec["adduct"].iloc[0]
        # candidates: widest window 20ppm expanded to >=300, hard cap CAND_CAP
        lo = bisect_left(smass_all, qmass - qmass * 20e-6)
        hi = bisect_right(smass_all, qmass + qmass * 20e-6)
        pcur = 20
        while hi - lo < 150 and pcur < 1000:
            pcur *= 2
            lo = bisect_left(smass_all, qmass - qmass * pcur / 1e6)
            hi = bisect_right(smass_all, qmass + qmass * pcur / 1e6)
        tcands = list(ssmi_all[lo:hi])
        clo = bisect_left(cmass, qmass - qmass * 20e-6)
        chi = bisect_right(cmass, qmass + qmass * 20e-6)
        ccands = list(csmi[clo:chi])
        seen, cands = set(), []
        for s in tcands + ccands:
            if s not in seen:
                seen.add(s); cands.append(s)
        if qs not in seen:
            cands.append(qs); seen.add(qs)
        # mass-sort, cap
        mall = np.array( [sm(s) for s in cands], dtype=float)
        order = np.argsort(np.abs(mall - qmass), kind="stable")
        cands = [cands[i] for i in order[:CAND_CAP]]
        n = len(cands)
        cm = np.array( [sm(s) for s in cands], dtype=float)

        # window spectra scoring (cos/ent) vs <=2 query spectra
        win = db_samp[db_samp["normalized_smiles"].isin(set(cands))]
        if len(win) > 3000:
            win = win.sample(3000, random_state=qi)
        qr = qspec.head(2)
        cosd, entd = {}, {}
        for _, r in qr.iterrows():
            for _, t in win.iterrows():
                try:
                    c = cosine(r["ms2_mzs"], r["ms2_normalized_intensities"],
                               t["ms2_mzs"], t["ms2_normalized_intensities"])
                    e = entropy_similarity(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                           t["ms2_mzs"], t["ms2_normalized_intensities"])
                except Exception:
                    c, e = 0.0, 0.0
                k = t["normalized_smiles"]
                if c > cosd.get(k, 0): cosd[k] = c
                if e > entd.get(k, 0): entd[k] = e
        cosv = np.array([cosd.get(s, 0.0) for s in cands], dtype=np.float32)
        entv = np.array([entd.get(s, 0.0) for s in cands], dtype=np.float32)

        # analog pool: db spectra within 200 Da, sample 1500, ent sim, top-200
        dm = np.abs(db_neu - qmass)
        idx = np.where(dm <= 200.0)[0]
        if len(idx) > 1500:
            idx = rng.choice(idx, 1500, replace=False)
        pool = db_samp.iloc[idx]
        scored = []
        for _, t in pool.iterrows():
            best = 0.0
            for _, r in qr.iterrows():
                try:
                    e = entropy_similarity(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                           t["ms2_mzs"], t["ms2_normalized_intensities"])
                except Exception:
                    e = 0.0
                if e > best: best = e
            if best > 0.05:
                scored.append((best, t["normalized_smiles"]))
        scored.sort(reverse=True)
        for K in (50, 100, 200):
            pass
        atop = scored[:200]
        asims = np.array([a[0] for a in atop], dtype=np.float32)
        assmi = [a[1] for a in atop]
        # tanimoto cand x analog (packed bits)
        Cpack = np.stack([pack_fp(fp[s]) if s in fp else np.zeros(32, np.uint64) for s in cands])
        Apack = (np.stack([pack_fp(fp[s]) if s in fp else np.zeros(32, np.uint64) for s in assmi])
                 if assmi else np.zeros((0, 32), np.uint64))
        T = tan_mat(Cpack, Apack) if len(assmi) else np.zeros((n, 0), np.float32)
        ana = {}
        for K in (50, 100, 200):
            m = min(K, T.shape[1])
            Tk = T[:, :m]; sk = asims[:m]
            for p in (3, 4):
                W = (sk ** p)[None, :]
                ana[f"{K}p{p}"] = (Tk * W).max(axis=1).astype(np.float32) if m else np.zeros(n, np.float32)

        # metfrag x3 tol (vectorized per cand)
        d = ADDUCT_DELTA.get(qadd) or 1.007276
        qm = np.asarray(qspec.iloc[0]["ms2_mzs"], dtype=float)
        qi_int = np.asarray(qspec.iloc[0]["ms2_normalized_intensities"], dtype=float)
        o = np.argsort(-qi_int)[:100]
        targets = np.sort(qm[o] - d)
        w = np.sqrt(np.maximum(qi_int[o], 0)); wsum = w.sum()
        o2 = np.argsort(qm[o] - d)
        ts, ws = targets, w[o2]
        met = {}
        for tol in (0.005, 0.01, 0.02):
            mv = np.zeros(n, dtype=np.float32)
            for i, s in enumerate(cands):
                fl = frags.get(s)
                if not fl or wsum <= 0:
                    continue
                fa = np.sort(np.asarray(fl, dtype=float))
                j = np.searchsorted(fa, ts)
                j = np.clip(j, 0, len(fa) - 1)
                j2 = np.clip(j - 1, 0, len(fa) - 1)
                hit = (np.abs(fa[j] - ts) <= tol) | (np.abs(fa[j2] - ts) <= tol)
                mv[i] = float((ws[hit]).sum() / wsum)
            met[tol] = mv

        # fp channel: predict query fp, dot + top512-dot
        import torch
        with torch.no_grad():
            X = np.stack([featurize_spec(r["ms2_mzs"], r["ms2_normalized_intensities"],
                                         r["adduct"], r["precursor_mz"],
                                         r["collision_energy_ev"]) for _, r in qr.iterrows()])
            Z = fpnet(torch.from_numpy(X)).numpy().mean(axis=0)
        F = np.stack([(fp[s].astype(np.float32) if s in fp else np.zeros(2048, np.float32)) for s in cands])
        fpdot = (F @ Z).astype(np.float32)
        top = np.argsort(-np.abs(Z))[:512]
        fptop = (F[:, top] @ Z[top]).astype(np.float32)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        zn = Z / (np.linalg.norm(Z) + 1e-9)
        fpcos = (Fn @ zn).astype(np.float32)

        # mass/formula features
        merr = np.abs(cm - qmass) / qmass
        lp = np.array([np.log1p(fprior.get(sform_tr.get(s, cform_map.get(s, "")), 0)) for s in cands],
                      dtype=np.float32)
        ik = np.array([sik14.get(s, "COCO:" + (cform_map.get(s, "?") or "?")) for s in cands], dtype=object)
        out.append(dict(cands=np.array(cands, dtype=object), cmass=cm, cos=cosv, ent=entv,
                        ana=ana, met=met, fpdot=fpdot, fptop=fptop, fpcos=fpcos,
                        merr=merr.astype(np.float32), lp=lp, ik=ik, truth=qs,
                        qmass=np.float32(qmass), qadd=qadd,
                        qnpeak=np.int32(len(qm)), winppm=np.abs(cm - qmass) / qmass * 1e6))
        if (qi + 1) % 20 == 0:
            print(f"  seed {seed} {qi + 1}/{len(queries)}", flush=True)
    np.savez_compressed(f"{CACHE_DIR}/seed{seed}.npz",
                        **{f"q{i}_{k}": (v if not isinstance(v, dict) else pickle.dumps(v))
                           for i, q in enumerate(out) for k, v in q.items()
                           if k != "cands"},
                        **{f"q{i}_cands": q["cands"] for i, q in enumerate(out)},
                        n=np.array([len(out)]))
    print(f"seed {seed} cached {len(out)} queries", flush=True)


def load_q(seed):
    z = np.load(f"{CACHE_DIR}/seed{seed}.npz", allow_pickle=True)
    n = int(z["n"][0])
    out = []
    for i in range(n):
        q = {}
        for k in ("cmass", "cos", "ent", "fpdot", "fptop", "fpcos", "merr", "lp",
                  "ik", "truth", "qmass", "qadd", "qnpeak", "winppm"):
            q[k] = z[f"q{i}_{k}"]
            if isinstance(q[k], np.ndarray) and q[k].dtype == object and q[k].size == 1:
                q[k] = q[k][0]
        q["cands"] = np.array(z[f"q{i}_cands"])
        q["ana"] = pickle.loads(bytes(z[f"q{i}_ana"].tobytes())) if f"q{i}_ana" in z else pickle.loads(z[f"q{i}_ana"])
        q["met"] = pickle.loads(bytes(z[f"q{i}_met"].tobytes())) if f"q{i}_met" in z else pickle.loads(z[f"q{i}_met"])
        out.append(q)
    return out


# ---------------- eval (no recompute) ----------------
def ranks_of(score):
    return np.argsort(np.argsort(-score, kind="stable"), kind="stable")


def fuse_rank(channels):
    R = np.stack([ranks_of(c) for c in channels], axis=1)
    return R.mean(axis=1)


def apply_dedup(cands, score, ik, mode):
    if mode == "exact":
        best = {}
        for i, s in enumerate(cands):
            if s not in best or score[i] > score[best[s]]:
                best[s] = i
        keep = sorted(best.values())
    else:
        best = {}
        for i, g in enumerate(ik):
            if g not in best or score[i] > score[best[g]]:
                best[g] = i
        keep = sorted(best.values())
    return keep


def mrr_combo(Q, spec, rank_cap=None, collect=None):
    """spec: dict(floor, fills[list of keys], dedup, window, cap, gbm_probs|None)."""
    rr = []
    for q in Q:
        cands = q["cands"]; n = len(cands)
        mask = np.ones(n, bool)
        if spec.get("window"):
            mask &= q["winppm"] <= spec["window"] + 1e-9
            # NO truth force-keep: tight windows must pay their recall cost
        idx = np.where(mask)[0]
        if len(idx) == 0:
            rr.append(0.0); continue
        floor = spec["floor"]
        base = {"cos": q["cos"], "ent": q["ent"],
                "mixed": (q["cos"] / (q["cos"].max() + 1e-9) + q["ent"] / (q["ent"].max() + 1e-9))}[floor]
        if spec.get("gbm") is not None:
            final = spec["gbm"]
        elif spec.get("fills"):
            ch = [base] + [FEATMAP[f](q) for f in spec["fills"]]
            # rerank top-W of floor base
            fb = ranks_of(base)
            top = np.argsort(fb, kind="stable")[:W_TOP]
            fr = fuse_rank([c[top] for c in ch])
            final = base.copy()
            final[top] = -fr  # lower mean-rank -> higher score; tail keeps floor order
            # merge: ranked by (in_top, score)
            order = np.concatenate([top[np.argsort(fr, kind="stable")],
                                    np.delete(np.arange(n), top)])
            rankarr = np.empty(n, int); rankarr[order] = np.arange(n)
            final = -rankarr.astype(float)
        else:
            final = base
        cap = spec.get("cap")
        if cap:
            pre = q["cos"] / (q["cos"].max() + 1e-9) + q["ent"] / (q["ent"].max() + 1e-9)
            keepc = set(np.argsort(-pre, kind="stable")[:cap])
            ii = np.array([i for i in idx if i in keepc])
            if len(ii):
                idx = ii
        keep = apply_dedup([cands[i] for i in idx], np.array([final[i] for i in idx]),
                           [str(x) for x in q["ik"][idx]], spec.get("dedup", "exact"))
        fidx = [idx[k] for k in keep]
        sord = np.argsort([-final[i] for i in fidx], kind="stable")
        ranked = [cands[fidx[k]] for k in sord]
        try:
            r = list(ranked).index(str(q["truth"])) + 1
        except ValueError:
            r = 10 ** 9
        if collect is not None:
            collect.append(r)
        rr.append(1.0 / r if r <= (rank_cap or MRR_CAP) else 0.0)
    return float(np.mean(rr))


def FEAT_cos(q): return q["cos"]
def FEAT_ent(q): return q["ent"]


FEATMAP = {}


def build_featmap():
    m = {"cos": lambda q: q["cos"], "ent": lambda q: q["ent"],
         "mixed": lambda q: q["cos"] / (q["cos"].max() + 1e-9) + q["ent"] / (q["ent"].max() + 1e-9),
         "fpdot": lambda q: q["fpdot"], "fptop": lambda q: q["fptop"], "fpcos": lambda q: q["fpcos"],
         "met005": lambda q: q["met"][0.005], "met01": lambda q: q["met"][0.01],
         "met02": lambda q: q["met"][0.02], "negmerr": lambda q: -q["merr"],
         "lp": lambda q: q["lp"]}
    for k in ("50p3", "50p4", "100p3", "100p4", "200p3", "200p4"):
        m[f"ana{k}"] = (lambda q, k=k: q["ana"][k])
    return m


FEATS31 = ["cos", "ent", "ana50p3", "ana50p4", "ana100p3", "ana100p4", "ana200p3",
           "ana200p4", "met005", "met01", "met02", "fpdot", "fptop", "fpcos",
           "negmerr", "ana_cos_gap", "cos_ent_gap", "ana_best", "met_best", "fp_gap",
           "cmass_n", "qmass_n", "qnpeak_n", "cos_top1", "ent_top1", "ana_top1",
           "met_top1", "fp_top1", "cos_mean", "ent_mean", "ncand_n"]
FSET = {"3": ["cos", "ent", "ana100p3"],
        "8": ["cos", "ent", "ana100p3", "met01", "fpdot", "negmerr", "ana_best", "fpcos"],
        "25": None, "31": None}
FSET["25"] = FEATS31[:25]
FSET["31"] = FEATS31[:]


def feat_matrix(Q, names):
    Xs, ys = [], []
    for q in Q:
        n = len(q["cands"])
        d = {}
        for k in ("cos", "ent", "ana50p3", "ana50p4", "ana100p3", "ana100p4",
                  "ana200p3", "ana200p4", "met005", "met01", "met02",
                  "fpdot", "fptop", "fpcos"):
            d[k] = FEATMAP[k](q).astype(float)
        d["negmerr"] = (-q["merr"]).astype(float)
        d["cos_ent_gap"] = d["cos"] - d["ent"]
        d["ana_best"] = np.maximum.reduce([d["ana50p3"], d["ana100p3"], d["ana200p3"]])
        d["ana_cos_gap"] = (d["ana_best"] - d["cos"]).astype(float)
        d["met_best"] = np.maximum.reduce([d["met005"], d["met01"], d["met02"]])
        mx = d["fpdot"].max()
        d["fp_gap"] = d["fpdot"] - np.sort(d["fpdot"])[::-1][min(1, n - 1)]
        d["cmass_n"] = (q["cmass"] / 1000.0).astype(float)
        d["qmass_n"] = np.full(n, float(q["qmass"]) / 1000.0)
        d["qnpeak_n"] = np.full(n, float(q["qnpeak"]) / 100.0)
        for kk, vv in (("cos_top1", "cos"), ("ent_top1", "ent"), ("ana_top1", "ana_best"),
                       ("met_top1", "met_best"), ("fp_top1", "fpdot")):
            v = d[vv]; t = np.sort(v)[::-1][min(4, n - 1)]
            d[kk] = (v - t)
        d["cos_mean"] = np.full(n, float(d["cos"].mean()))
        d["ent_mean"] = np.full(n, float(d["ent"].mean()))
        d["ncand_n"] = np.full(n, n / 2000.0)
        X = np.stack([np.nan_to_num(d[k], nan=0.0, posinf=0.0, neginf=0.0) for k in names], axis=1)
        y = (np.array([str(s) for s in q["cands"]]) == str(q["truth"])).astype(int)
        Xs.append(X); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("cache", "all"):
        # fix struct_mass via closure over loaded frames
        tr, cf, fp, frags = load_static()
        coco_med = dict(zip(cf["canonical_smiles"], cf["exact_molecular_weight"]))
        sform_all = tr.groupby("normalized_smiles")["molecular_formula"].first().to_dict()
        cform_map = dict(zip(cf["canonical_smiles"], cf["molecular_formula"])) if "molecular_formula" in cf.columns else {}
        net = fp_model()
        for s in SEEDS:
            cache_seed(s, tr, cf, fp, frags, net, coco_med, sform_all, cform_map)
    if stage in ("eval", "gbm", "report", "all"):
        globals()["FEATMAP"] = build_featmap()
        print("eval/GBM stages run via run_eval.py", flush=True)
