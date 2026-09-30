# Self-audit: our modules (adversarial, execution-verified)

Scope: v10/submit_v10.py, v11/submit_v11.py + enrich_train.py, v13/frag_up.py + enrich_bde.py,
v2/train_fp.py + blend.py, v4/channels.py, v5/validate_v5.py + feat_cache builders,
v7/submit_v7.py + metfrag.py, v12/compare_all.py + v14/stack_bout.py (read-only).
Method: full read + `grep` for silent-failure patterns + execution on real data
(hit rates below are measured, not guessed). No fixes applied.

## Production-impact order

### 1. v13/enrich_bde.py — BUG (HIGH)
- `ladder_score` ion grid (`enrich_bde.py:29`) builds
  `frags + dh*1.007825 + 1.007276` for `dh in (-2..+2)`. Frags from
  `frag_up.fragment_masses_bde` already contain `w` and `w+1.007825` variants,
  so the grid double-adds the H-rearrangement AND sweeps ±2.015 Da on top of the
  ±0.01 Da tolerance. Measured on a real train spectrum: ladder 0.420 vs strict
  `frag+1.007276` 0.027 (~15x inflation). The validated "BDE" GBM variants
  (`gbm-bde`, `gbm-both`) train on this inflated score, so their deltas vs
  `gbm-oldfrag` are not pure physics gains.
  Fix: score `frags + adduct_delta` only (per-adduct delta, no dh sweep), or
  report both strict and sweep as separate features.
- Positional cache join (`enrich_bde.py:47-52`): `d["bde"] = extra["bde"].values`
  with no key check; `feat_bde_seed*.parquet` stores only `[bde]`. Lengths match
  today (28474==28474) but any v5-cache regeneration/reorder silently misaligns.
  The keyed variant (`feat_bde_keyed*.parquet` with seed/truth/cand) exists —
  use it with a merge, assert row equality.
- Silent dead branch (`enrich_bde.py:60-65`): `except KeyError: vals.extend(0..)`
  never fires on frozen cache (0/150 truths missing, measured); `fillna(0)`
  (`enrich_bde.py:80`) then maps any residual NaN bde to worst — correct
  direction, but masks upstream misses.

### 2. v7/metfrag.py — BUG (HIGH, feeds v11/v12/v13 metfrag + frag_cache.pkl)
- Bond-index domain error (`metfrag.py:20-31`): bonds enumerated on the
  H-added mol (`Chem.AddHs`, 21 bonds for aspirin) but applied to the H-stripped
  mol (`RemoveHs`, 13 bonds) via `FragmentOnBonds(nm, combo)`. Measured: H-added
  has 13 single non-ring bonds with max idx 20; no-H mol has 13 total bonds and
  only 5 single non-ring bonds. Out-of-range/wrong-index combos raise inside the
  `try` and are skipped, so the cache enumerates the wrong bond set yet still
  returns fragments (30 for aspirin) — consistently wrong in train AND inference
  (rank-consistent, physically wrong). Fix: enumerate bonds on `nm` directly.
- Silent defaults (`metfrag.py:64`): `ADDUCT_DELTA.get(adduct, 1.007276)` —
  0% trigger on test (all 7 test adducts known) but nonzero on train dimers/
  exotic adducts; wrong-delta matches score as if +H. Fix: return 0.0 or NaN-flag
  on unknown adduct instead of +H default.
- `frag_score` top_n=100 without intensity floor differs from v10 `denoise`
  (floor 0.01) and v14 `_clean` (floor 0.002, top 256) — same channel, three
  cleanings; rank-consistent within each pipeline but cross-pipeline numbers
  (v12 vs v14) are not apples-to-apples.

### 3. v2/train_fp.py — BUG (MEDIUM-HIGH, production floor model input)
- `meta_vec` NaN path (`train_fp.py:29-30`): `float(np.mean(list(ce_list)))`
  yields NaN for empty-list CE (verified `meta_vec(...,[]) -> [..., nan]`).
  Hit rate today: 0 empty-lists in 2.54M train + 1213 test (all empties are
  `None` → 0.0, correct), so latent, not live. Fix: `if not ce_list: ce=0.0`.
- Architecture drift (dead reproduction path): `train_fp.main` builds
  `FpMLP()` with default `d_h=1024`, but production `data/fp_trans.pt` is
  `d_h=1536` (1536×12012 weight matrix, verified) while `v2/fp_mlp.pt` is
  1024-wide. Re-running `v2.train_fp` cannot reproduce production weights
  (load would fail on shape). Fix: pin `d_h=1536` + record seed/epochs, or
  document fp_trans.pt provenance.
- One-hot coverage: `ADDUCTS` (10 entries) misses 13.8% of train spectra
  (all dimers `[2M+*]`, `[M-2H2O+H]+`, `[M+2H]2+`, …) → all-zero adduct block,
  silently. Test is 0% (all 7 test adducts covered). Fix: add dimer/2+ flags or
  a dedicated unknown-adduct bit.
- `bin_spectrum` truncation (`train_fp.py:15-17`): m/z > 1200 silently dropped
  by the `ok` mask. Measured 6/20000 spectra (0.03%) have peaks above 1200
  (max 1737). Negligible but unlogged. Fix: log drop rate or raise MZ_MAX.

### 4. v11/enrich_train.py — BUG (MEDIUM, production ranker training)
- `qmass` is median-CANDIDATE-mass, not query mass (`enrich_train.py:94-96`):
  `feat["qmass"] = groupby(seed,truth)["cmass"].transform("median")`, then
  `mass_err = |cmass - qmass|/qmass`. Inference (`submit_v11.py:166`) uses
  `|cmass - query_neutral|/query_neutral`. Conceptually wrong; numerically
  near-harmless because pools are mass-windowed (median |median_cand-true|/true
  = 1.5e-5, max 4.1e-4 over seed-10 truths, measured). Fix: store true query
  neutral per (seed,truth) from train precursor instead of candidate median.
- `fillna(0)` direction error (`enrich_train.py:110`): `feat[FEATS].fillna(0)`
  maps missing `mass_err` (NaN from NaN cmass/qmass) to 0.0 = perfect mass
  match. Correct worst-case is 1.0 (as inference uses). No live trigger today
  (`_smas` covers all train+coconut via setdefault), but the guard rewards
  missing data if it ever fires. Fix: `fillna({"mass_err": 1.0, ...0})`.
- `t_top1` loop (`enrich_train.py:99-109`): `if not tv: continue` leaves
  `t_top1=0.0` (fine), but per-row `tfp.get` miss also leaves 0.0 while the
  analog ranking that selected `top` used full fps — consistent. Minor: chained
  `feat.at` writes in a loop are slow, not wrong.

### 5. v13/frag_up.py — BUG (MEDIUM, validated BDE numbers rest here)
- Pseudo-distance (`frag_up.py:79`): `min(abs(bond_idx - catom_idx)+...)`
  uses atom INDEX arithmetic, not graph distance. Correlated on linear chains,
  wrong on branched/fused systems (charge-proximal ordering misranks).
  Fix: BFS graph distance via `Chem.GetDistanceMatrix` or neighbor walk.
- Redundant fallback (`frag_up.py:50`):
  `_BDE.get(key, _BDE.get((key[1],key[0]), 350))` — `key` is already sorted so
  the inner get never adds coverage; every unseen pair (P-O, N-O, N-S, O-S,
  C-Si/B/Se…) scores 350. Measured P-O → 350. Not silent-wrong for ORDERING
  (documented "rough"), but extend the table or log fallback rate.
- Adduct default (`frag_up.py:119`): `.get(adduct, 1.007276)` — dimer/exotic
  adducts scored as +H. Same fix as metfrag: explicit unknown handling.
- Filter asymmetry: only `SINGLE` bonds cleaved (`frag_up.py:60-61`) — aromatic
  bonds get a 475 BDE entry but are excluded by the filter, so fused aromatics
  fragment only via substituents (measured phenylacetate: 3/10 bonds kept).
  By design, but the `_BDE` aromatic entry is then dead code — document it.
- Tolerance inconsistency: frag match ±0.01 Da fixed (`frag_up.py:136`) vs
  neutral-loss ladder 15 ppm (`frag_up.py:146`); at m/z 100, 0.01 Da = 100 ppm.
  Fix: use ppm in both or document why fixed-Da is intended.

### 6. v11/submit_v11.py — BUG (MEDIUM, production ranker path)
- Dead duplicate (`submit_v11.py:68` then `:82`): `_tneut` assigned twice from
  the same frame — harmless, remove one.
- `_frags.get(s, [])` (`submit_v11.py:161`) + `frag_match(..., []) → 0.0`:
  in-window miss measured 4.5% train / 0.1% coconut (cache was built window-aware;
  global coverage is far lower: 33% train / 14% coconut). Documented fallback,
  correctly worst-case.
- `mass_err` guard (`submit_v11.py:166`): `... if _cm==_cm else 1.0` — correct
  direction; dead today (`_smas` covers train+coconut fully). Keep.
- Analog subsample variance (`submit_v11.py:123-126`): pool within ±200 Da is
  ~930k rows, then `.sample(2000, random_state=mi)` (0.2%). Deterministic per
  molecule-index but high-variance vs training (`validate_v5` uses
  `random_state=qi`, different key) — train/inference analog-distribution shift
  by construction. Fix: shared seeded sampler or larger pool + note variance.
- `_fprior`/`_sform` miss → `log1p(0)=0` (`submit_v11.py:167`): 36% of coconut
  formulas score 0 prior (measured) — correct (unseen = rare), just note the
  train-only prior penalizes novel chemistry.

### 7. v10/submit_v10.py — BUG (MEDIUM, production floor)
- Fixed ±10 ppm window (`submit_v10.py:38-41`, floor 0.01 Da) vs expanding
  window elsewhere: measured 219 train cands/query (min 2, max 621, 0 empty over
  400 test mols) vs expanding-window 319 (min 195). Tighter = cleaner but
  recall-capped if precursor error exceeds 10 ppm; train precursor error median
  is 1.0 ppm but tail reaches 5e5 ppm (measured). No live empty window on test,
  but no `min_n` guard like v5/v7. Fix: keep floor but add min_n expansion or
  assert non-empty per mol.
- `tcands[:1500]` cap (`submit_v10.py:109`) is dead (max 621 observed) — harmless.
- `tfp.get(s, cfp.get(s)) ... continue` (`submit_v10.py:110-112`): skip rate ~0
  (train ⊂ tfp, coconut ⊂ cfp, both 100% key coverage measured). Keep the skip
  but log it.
- Dimer/exotic neutral NaN (`submit_v10.py:23-24`): 8516/2.54M train (0.34%)
  dropped via `isfinite`; test 0/1213. Correct direction, unlogged per-adduct.

### 8. v12/compare_all.py — BUG (MEDIUM, channel-comparison integrity)
- `exp` channel handicapped (`compare_all.py:33-36,52-54`): formula map built
  from TRAIN only, so all coconut candidates (14639 measured in seed-10 cache)
  score `exp=0.0` via `if fmap.get(s) else 0.0`. The published channel ranking
  (cos/ent/ana/exp/metfrag/fpdot on "identical" candidates) systematically
  depresses `exp`. Fix: map coconut formulas from `coconut_fp.parquet`.
- Single-spectrum query (`compare_all.py:44`): `rows.iloc[0]` while
  enrich_train/bout use `head(2)` mean — inconsistent query representation
  across the numbers being compared. Fix: use the same 2-spectrum mean.
- Silent skips (`compare_all.py:40-43`): `except KeyError: continue` never fires
  on frozen cache (0/150 missing) but would silently drop queries on cache
  refresh. Fix: raise or count.

### 9. v7/submit_v7.py — BUG (LOW-MEDIUM)
- `AD.get(adduct, 1.007276)` (`submit_v7.py:50`): same unknown-adduct default
  as metfrag; test 0% trigger. Fix: shared adduct module with explicit unknown.
- `topn` without intensity floor or sort (`submit_v7.py:30-34`): keeps top-150
  by intensity but returns UNSORTED m/z; `cosine` re-sorts internally so
  correct, but `frag_match` also re-sorts by intensity — consistent yet
  divergent from v10 `denoise` (floor 0.01 + sort). Cross-version cosine numbers
  are not strictly comparable. Document the cleaning per version.
- Fills skip (`submit_v7.py:110-111`): `if f is None: continue` drops
  candidates with no precomputed frags (~4.5% train-window, measured) rather
  than scoring 0 — changes fills depth per query. Prefer scoring 0 to keep
  candidate depth uniform.

### 10. v5/validate_v5.py + feat_cache builders — SUSPECT (LOW-MEDIUM)
- `asims`/`afps` slice misalignment (`validate_v5.py:104-105`,
  mirrored in `submit_v5.py:99-100`):
  `afps=[... if s in tfp]; asims=[a[0] for a in analogs[:len(afps)]]` pairs the
  first-N sims with the filtered fps. Live trigger 0% (all train analog structs
  are in tfp, measured `train ⊂ tfset`), so latent. v11's `_an/_af` co-append
  pattern is the correct template — backport it.
- Stale-cache early return (`validate_v5.py:38-40`): once
  `v5/feat_cache/seed*.parquet` exists the compute branch NEVER executes, so
  all downstream numbers (v11/v12/v13/v14) silently pin to the cached 50-query
  sample. Process risk, not math bug. Fix: hash code+config into cache key or
  assert freshness.
- Sampling-key drift: validate pools `.sample(1200, random_state=qi)` vs
  submit `.sample(2000, random_state=mi)` — train/inference analog pools differ
  in size AND seed key. Note as variance source.

### 11. v5/submit_v5.py — SUSPECT (LOW, inherits v5 builder patterns)
- Same `afps`/`asims` slice pattern (`submit_v5.py:99-100`), 0% live trigger
  (same train⊂tfp argument). Fix identically.
- Analog pool `.sample(2000, random_state=mi)` (`submit_v5.py:89`) vs training
  `random_state=qi`: same drift note as v11. Deterministic per run, but
  molecule-index seeding couples sampling to test order.

### 12. v2/blend.py — CLEAN-ish (LOW)
- Threshold inconsistency: blend ranks with `pred>0.3` (`blend.py:109`) while
  `train_fp` validates at 0.5 (`train_fp.py:98-99`). Different operating points
  for memory- vs learned-fp paths — document, not a bug in either alone.
- `den==0 → pred=None → t=0.0` for ALL cands (`blend.py:104-110`): full-tie
  fallback measured 0/5 probe queries (den 0.5–14.9, 85–909 sims>0.01). Latent;
  tie-break is arbitrary order. Fine with a log.
- `head(3)` db spectra (`blend.py:70`) vs `head(2)` everywhere else — sample
  depth differs across validation numbers. Note only.

### 13. v4/channels.py — CLEAN (LOW, by-design guards verified)
- `<3 matched peaks → 0.0` (`channels.py:48-49`): fires on 156/200 random train
  pairs (78%) — this is the documented selectivity mechanism, not a silent bug.
- `_entropy` zero-guard (`channels.py:37`), tanimoto union-guard
  (`channels.py:59`), `clean` empty-guard (`channels.py:12`): all correct
  directions; empty rates 0/2000 spectra. No fix.

### 14. v14/stack_bout.py — READ-ONLY audit, no behavior touched (LOW for production)
- CE fallback (`stack_bout.py:147`): unparseable/empty CE → 25.0. Deterministic;
  note as model-input prior, not bug.
- Missing-fp zeros (`stack_bout.py:167`): `np.zeros(6930)` per missing cand —
  keeps matrix shape; rank-features then compare zeros vs logits (all-zero rows
  tie). Coverage-dependent; log miss rate per bout.
- Missing-frag zeros (`stack_bout.py:172`): `.get(s, zeros(0)) → explain→0.0`.
  Correct worst-case direction.
- `corrcoef(lv,-mr)` guard (`rank.py:57`) checks only `lv.std()>1e-9`; if `mr`
  is constant (e.g. all-zero cfp rows) the coefficient is NaN (verified) and
  propagates as a constant-NaN feature (HGB treats NaN as missing — silent but
  contained). Fix (not applied): guard `mr.std()` too.
- Broad `try/except: pass` around merged-peak logits (`stack_bout.py:162-163`)
  and frag precompute (`stack_bout.py:43-44`): fallback = single-spectrum
  logits / empty frags. Acceptable, but count fires.
- `md` flag (`stack_bout.py:175`): `endswith("]+") → 1.0 else -1.0` is a coarse
  ion-mode proxy passed to `their_explain`; dimers/multimers share the +H branch.
  Their-side modeling choice — do not touch without their spec.

### 15. v5/rerank_gbm.py — CLEAN
- Straightforward 3-seed HGB over cached cos/ent/ana; `mrr` helper truncates at
  25 with 1e9-miss convention shared across files (consistent). Final refit on
  all seeds then dumps `gbm_ranker.pkl` — standard, no silent path. Only note:
  inherits the stale-cache pinning from `validate_v5.build_seed`.

## Arithmetic spot-checks (verified by execution)
- Dimer neutral masses (`(prec∓H)/2`) correct in all submit files; `[M-H2O+H]+`
  delta −17.003348 → `prec+17.003` correct.
- `window10` floor `max(q*10/1e6, 0.01)`: at m/z 100 the floor (0.01 Da =
  100 ppm) dominates the 10 ppm claim — documented hybrid, keep the name honest.
- `frag_up` NL menu masses (H2O/CO/NH3/HCOOH/CO2/hexose…) match monoisotopic
  values; NL ladder `prec-loss` in ion space is dimensionally correct.
- `enrich_bde.H=1.007825` vs adduct proton 1.007276: the 1.5 mDa electron-mass
  gap is fine for strict matching but is dwarfed by the ±2 Da dh sweep above.

## Join/merge key summary
- Fingerprint namespaces are disjoint by construction: train⊂`fingerprints`
  (100% overlap, column named `smiles` but values are normalized — verified),
  coconut⊂`coconut_fp` (0 overlap with train — verified). `tfp.get(s,
  cfp.get(s))` is therefore correct; `blend.load_fp` (train-only) must never see
  coconut cands (it doesn't — blend is train-pool only).
- Only positional (key-less) join found: `enrich_bde` bde assignment (see §1).
  All other merges are keyed or same-frame positional (safe).
