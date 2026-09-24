"""v-grid eval: combo grid on cached channels, no recompute. GBM LOSO. Report."""
import os, sys, pickle, importlib.util, numpy as np

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
spec = importlib.util.spec_from_file_location("vg", f"{PROJECT}/v-grid/run_grid.py")
vg = importlib.util.module_from_spec(spec)
sys.argv = ["vg"]
spec.loader.exec_module(vg)
vg.FEATMAP = vg.build_featmap()

SEEDS = (10, 11)
SHIP = 0.012

Q = {s: vg.load_q(s) for s in SEEDS}
print({s: len(Q[s]) for s in SEEDS}, flush=True)
BASE = dict(floor="cos", fills=["ana100p3"], dedup="exact", window=None, cap=None)


def ev(name, **kw):
    spec = dict(BASE)
    spec.update(kw)
    per, diag = {}, {}
    for s in SEEDS:
        col = []
        per[s] = vg.mrr_combo(Q[s], dict(spec), collect=col)
        col = np.array(col, float)
        diag[s] = (float(np.mean(col <= 100)), float(np.median(col)))
    return name, spec, per, diag


rows = []
rows.append(ev("anchor cos-only", floor="cos", fills=[]))
rows.append(ev("floor ent-only", floor="ent", fills=[]))
for k in ("50p3", "50p4", "100p3", "100p4", "200p3", "200p4"):
    rows.append(ev(f"ana{k}", fills=[f"ana{k}"]))
for w in (8.5, 10.0, 20.0):
    rows.append(ev(f"window{w}ppm", window=w))
rows.append(ev("cap2000", cap=2000))
rows.append(ev("metfrag tol0.005", fills=["ana100p3", "met005"]))
rows.append(ev("metfrag tol0.01", fills=["ana100p3", "met01"]))
rows.append(ev("metfrag tol0.02", fills=["ana100p3", "met02"]))
for f in ("fpdot", "fptop", "fpcos"):
    rows.append(ev(f"fp-{f}", fills=["ana100p3", f]))
for fl in ("ent", "mixed"):
    rows.append(ev(f"floor-{fl}+ana", floor=fl))
rows.append(ev("dedup-ik14", dedup="ik14"))
rows.append(ev("all-on cos", fills=["ana100p3", "met01", "fpdot"]))
rows.append(ev("all-on mixed", floor="mixed", fills=["ana200p4", "met01", "fpdot"]))
rows.append(ev("all-on ent", floor="ent", fills=["ana200p4", "met01", "fpdot"]))
rows.append(ev("no-met", fills=["ana100p3", "fpdot"]))
rows.append(ev("no-fp", fills=["ana100p3", "met01"]))

# window recall diagnostic (truth passes window on its own mass; signed bug fixed with abs)
for w in (8.5, 10.0, 20.0):
    for s in SEEDS:
        strict = np.mean([abs(float((q["cmass"][np.array(list(map(str, q["cands"]))) == str(q["truth"])][0]) - float(q["qmass"])) / float(q["qmass"]) * 1e6) <= w + 1e-9 for q in Q[s]])
        print(f"recall win{w}: seed{s} truthpass={strict:.3f}", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier
gbm_specs = {}
for fs, names in vg.FSET.items():
    probs = {}
    for held in SEEDS:
        trn = [s for s in SEEDS if s != held][0]
        Xtr, ytr = vg.feat_matrix(Q[trn], names)
        rng = np.random.default_rng(held)
        pos = np.where(ytr == 1)[0]
        neg = np.where(ytr == 0)[0]
        per_q = 60
        nq = len(Q[trn])
        keep_neg = rng.choice(neg, min(len(neg), per_q * nq), replace=False)
        keep = np.concatenate([pos, keep_neg])
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                             max_leaf_nodes=15, l2_regularization=10.0,
                                             random_state=held)
        clf.fit(Xtr[keep], ytr[keep])
        Xh, _ = vg.feat_matrix(Q[held], names)
        probs[held] = clf.predict_proba(Xh)[:, 1]
        with open(f"{PROJECT}/v-grid/gbm_F{fs}_held{held}.pkl", "wb") as fh:
            pickle.dump({"model": clf, "feats": names}, fh)
    gbm_specs[fs] = probs

# split flat prob vectors back per query for mrr_combo via gbm key: precompute per-query arrays
gbm_q = {fs: {s: [] for s in SEEDS} for fs in vg.FSET}
for fs in vg.FSET:
    for held in SEEDS:
        names = vg.FSET[fs]
        # recompute row counts per query
        off = 0
        p = gbm_specs[fs][held]
        for q in Q[held]:
            n = len(q["cands"])
            gbm_q[fs][held].append(p[off:off + n])
            off += n

for fs in vg.FSET:
    per = {}
    for s in SEEDS:
        rr = []
        # temporarily attach gbm scores: mrr_combo reads spec['gbm'] as full vector; do per-query here
        for q, g in zip(Q[s], gbm_q[fs][s]):
            qq = dict(q)
            n = len(q["cands"])
            spec = dict(BASE, gbm=None)
            # emulate: final = g within dedup exact, window/cap none
            keep = vg.apply_dedup(list(q["cands"]), g, [str(x) for x in q["ik"]], "exact")
            sord = np.argsort([-g[i] for i in keep], kind="stable")
            ranked = [str(q["cands"][keep[k]]) for k in sord]
            try:
                r = ranked.index(str(q["truth"])) + 1
            except ValueError:
                r = 10 ** 9
            rr.append(1.0 / r if r <= vg.MRR_CAP else 0.0)
        per[s] = float(np.mean(rr))
    dg = {}
    for s in SEEDS:
        col = []
        for q, g in zip(Q[s], gbm_q[fs][s]):
            keep = vg.apply_dedup(list(q["cands"]), g, [str(x) for x in q["ik"]], "exact")
            sord = np.argsort([-g[i] for i in keep], kind="stable")
            ranked_c = [str(q["cands"][keep[k]]) for k in sord]
            try:
                r = ranked_c.index(str(q["truth"])) + 1
            except ValueError:
                r = 10 ** 9
            col.append(r)
        col = np.array(col, float)
        dg[s] = (float(np.mean(col <= 100)), float(np.median(col)))
    rows.append((f"GBM-F{fs}", dict(floor="cos", fills=["gbm"], dedup="exact"), per, dg))

anchor = rows[0][2]
anchor_diag = rows[0][3]
print("\nranked (by mean MRR):", flush=True)
ranked = sorted(rows, key=lambda r: -(r[2][10] + r[2][11]) / 2)
for name, spec, per, dg in ranked:
    m = (per[10] + per[11]) / 2
    d = [(per[s] - anchor[s]) for s in SEEDS]
    ship = "SHIP" if all(x >= SHIP for x in d) else ("REGRESS" if all(x <= -SHIP for x in d) else "hold")
    print(f"{m:.4f} {name:22s} s10={per[10]:.4f} s11={per[11]:.4f} "
          f"d10={d[0]:+.4f} d11={d[1]:+.4f} {ship} "
          f"hit100={dg[10][0]:.2f}/{dg[11][0]:.2f} medr={dg[10][1]:.0f}/{dg[11][1]:.0f}", flush=True)

with open(f"{PROJECT}/v-grid/grid_table.txt", "w") as fh:
    for name, spec, per, dg in ranked:
        m = (per[10] + per[11]) / 2
        d = [(per[s] - anchor[s]) for s in SEEDS]
        ship = "SHIP" if all(x >= SHIP for x in d) else ("REGRESS" if all(x <= -SHIP for x in d) else "hold")
        fh.write(f"{m:.4f} {name:22s} s10={per[10]:.4f} s11={per[11]:.4f} "
                 f"d10={d[0]:+.4f} d11={d[1]:+.4f} {ship} hit100={dg[10][0]:.2f}/{dg[11][0]:.2f} "
                 f"medr={dg[10][1]:.0f}/{dg[11][1]:.0f} {spec}\n")
print("wrote v-grid/grid_table.txt", flush=True)
