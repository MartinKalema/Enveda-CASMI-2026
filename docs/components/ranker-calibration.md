# Ranker calibration: the fork-034 31-feature GBM ensemble

Source: `kernels/fork-034/notebook.ipynb` cell 8 (`rank_features`, ranker fit,
`rank_proba`), config in cell 1 (`CFG`), inference in cell 9. Upstream context:
`docs/engine-deep-dive.md` §1 (ranker), §2.5; `docs/kaggle-notebooks-survey.md`
findings 7, 10–11. Baseline: LB 0.328 fork of the top solution.

Training data: precomputed `rank_train.npz` (`X`, `Y`, `M`) from dataset
`prvsiyan/casmi26-ranker-features` — built offline on train molecules with the
same `rank_features` code. `Y` = 1 if candidate is the true structure.
`M == 0` marks Class-1 rows (query has a same-structure reference spectrum);
all other rows are Class-2. Sample weight is `W = where(M == 0, w1, 1 - w1)`.

## 1. The 31 features, one by one

Notation per query molecule: `lv` = per-candidate library entropy sim (Class-1
channel, 0 when absent); `ap` = analog score `max_a sim(a)^4 · Tanimoto`
(`P_SIM = 4.0`, `N_ANALOG = 100`); `raw = f·z` = candidate fingerprint dotted
with FPNet logits (exact Bayes LL); `nrm = raw/sqrt(cs)` = bit-count
normalised variant; `fr` = MetFrag-lite explained-intensity fraction;
`nc` = candidates in the ±8.5 ppm window. `_rank_norm` = rank/(nc−1) in
[0,1] (0 = best); `_z` = within-query z-score. All rank/z/gap features are
computed **within the query's candidate list**, so each row carries listwise
context into a pointwise classifier (this is the central trick — see §3).

### Library block (cols 0–4): the Class-1 witness

| # | Feature | Evidence carried |
|---|---------|------------------|
| 0 | `lv` (raw) | Absolute direct-match strength. Near 1 ≈ query spectrum seen in library. |
| 1 | `rank_norm(lv)` | Relative standing. Lets the GBM learn "best of a bad lot" vs "dominated". |
| 2 | `lvmax` (query constant) | Query-level context: is this a Class-1-type query at all? Gates the whole block. |
| 3 | `lv - lvmax` (gap) | Distance to the leader. 0 = leader; strongly negative = also-ran. Splits cleanly. |
| 4 | `(lv > 0)` hit flag | Whether any library evidence exists. Hard switch for "ignore this block". |

### Analog block (cols 5–13): the Class-2 workhorse

| # | Feature | Evidence carried |
|---|---------|------------------|
| 5 | `ap` = `max(tan·w⁴)` | Primary Class-2 score. `w⁴` crushes weak shifted matches that flood a plain max. |
| 6 | `rank_norm(ap)` | Relative analog standing among same-mass isomers. |
| 7 | `apmax` (query constant) | How good is the best analog explanation available for this query. |
| 8 | `ap - apmax` | Gap to analog leader. |
| 9 | `a1` = `max(tan·w¹)` | Unsharpened variant. Pair (ap, a1) tells the GBM whether the leader survives sharpening (robust scaffold hit) or only appears under p=1 (diffuse weak support). |
| 10 | `best_tan` = `max(tan)` | Pure structural proximity to the nearest analog, unweighted by spectral sim. High tan + low ap = "close structure, dissimilar spectrum" (suspicious). |
| 11 | `top_tan` = tan to top-sim analog | Structure of the single most spectrally-trusted relative. Differs from best_tan when the best spectral match and nearest structure disagree. |
| 12 | `mean_tan` (w⁴-weighted mean) | Breadth of support: one lucky analog vs a whole scaffold family pointing here. |
| 13 | `top_sim` (query constant) | Trust in the analog set itself. Low = all shifted matches are weak, discount cols 5–12. |

### Query-shape feature (col 14)

| # | Feature | Evidence carried |
|---|---------|------------------|
| 14 | `log(nc)` (query constant) | Candidate multiplicity. Large nc = crowded mass window (many isomers); the GBM learns to demand stronger evidence / flatten posteriors there. Also absorbs train/test window-density shift. |

### Model block (cols 15–20): the neural fingerprint in three views

`raw = f·z` is unbounded (grows with bit count); `nrm` corrects that. For each
of the two scalings the code gives three views:

| # | Feature | Why this view |
|---|---------|---------------|
| 15/18 | `_z(raw)`, `_z(nrm)` | Standardised margin: how many std above the query mean. Comparable across queries with different nc/score scales. |
| 16/19 | `_rank_norm(raw)`, `_rank_norm(nrm)` | Pure ordinal signal. Immune to logit miscalibration across spectra — the model can be overconfident globally and still rank correctly. |
| 17 | `raw - raw.max()` | Gap to model leader in nats (log-likelihood units). 0 = model favourite; −50 = decisively rejected. Keeps the absolute margin that rank discards. |
| 20 | `(raw == raw.max())` flag | Model-argmax indicator. Lets trees split "model's top pick" without re-deriving it. |

Why raw/z/rank of the **same** score all help: they are different projections
with different failure modes. Raw/gap preserve calibrated margins (a 40-nat gap
means more than a 2-nat gap); z normalises for query difficulty; rank survives
monotone distortions (per-spectrum logit scale shifts, merged-vs-single view
averaging). GBM splits are axis-aligned, so handing it all three views lets it
pick "rank < 0.1 AND gap > −5" conjunctions no single view expresses. Cost is
nil (same matmul). The same logic motivates the ap/a1 pair and lv/rank/gap
triples: correlated features are not redundant for trees, they are alternative
split geometries.

### Fragment block (cols 21–24)

| # | Feature | Evidence carried |
|---|---------|------------------|
| 21/22/23/24 | `fr`, `rank(fr)`, `fr - fr.max()`, `_z(fr)` | Physics channel: fraction of sqrt-intensity explainable by 1–2-bond cleavage. Same raw/rank/gap/z grammar as the model block (minus the argmax flag). Near-orthogonal to analog (true-score corr ≈ 0.058), so it stacks. |

### Agreement block (cols 25–30): corroboration logic as features

`mr` = rank_norm of model raw (0 = model favourite). `lbest`/`abest` = argmax
of lv/ap. `agree = 1 − mr[lbest]` = "how highly does the model rank the library
favourite" (1 = model agrees it is top; 0 = model buries it). `agree_a` = same
for the analog favourite.

| # | Feature | Corroboration logic |
|---|---------|---------------------|
| 25 | `lv·(1−mr)` (per-candidate) | Library evidence **discounted by model disagreement**. High only when both channels point here — the AND gate as a number. |
| 26 | `ap·(1−mr)` (per-candidate) | Same for analog: scaffold support that the neural model also likes. |
| 27 | `agree` (query constant) | Global Class-1 trust switch. High = library and model concur → trust lib. Low = library hit is a lone wolf (typical Class-2 trap) → ignore lib. |
| 28 | `agree_a` (query constant) | Global analog trust switch. High = two independent channels (spectral relatives + learned substructures) converge. |
| 29 | `agree·lvmax` (query constant) | Interaction term: corroborated **and** strong. Separates "weak-but-agreed" from "strong-and-agreed". |
| 30 | `corr(lv, −mr)` (query constant) | Whole-list rank correlation between library and model orderings. Positive = channels tell the same story (Class-1 regime); ~0/negative = library ordering is noise (Class-2 regime) → the GBM learns to zero the lib block. |

## 2. Why calibration beats hand-weighting

Measured collapse (upstream report, `docs/kaggle-notebooks-survey.md` finding
7): adding library similarity with a **fixed weight** to an analog-only scorer
drops Class-2 MRR **0.52 → 0.27**. Mechanism: for a Class-2 query the true
structure has no reference spectrum, so *every* Class-1 hit is a wrong
same-mass isomer — yet library sims on wrong isomers routinely reach 0.5–0.8
(shared core fragments), outscoring the true candidate's analog evidence. A
fixed weight cannot distinguish "lib hit corroborated" from "lib hit
accidental" because that distinction lives in the **joint** distribution
(model rank of the lib favourite, list correlation), not in either score alone.

The GBM learns exactly that joint rule through the agreement block: trust lib
when `agree`/correlation are high, else back off to analog/model/frag. This is
calibration in the ranking sense — mapping incommensurable channel scores
(entropy sims in [0,1], logit dots in nats, explained fractions) onto one
comparable posterior per query — and no linear blend can express the gating.

**Class asymmetry.** Class-1 queries are decided almost entirely by cols 0–4 +
agreement (library MRR ≈ 0.87–1.0); Class-2 queries must actively *suppress*
cols 0–4 and decide on cols 5–30 (analog MRR ≈ 0.52–0.61). The two regimes need
opposite treatment of the same features, which is why:

- Sample weighting `W1` matters: `W = where(M == 0, w1, 1 − w1)` rebalances
  Class-1 vs Class-2 rows. The fork fits **two priors, w1 ∈ {0.30, 0.60}**,
  and averages — hedging the Class-1/Class-2 tradeoff instead of betting on
  one operating point (measured +0.003 LB vs a single prior). Note w1 is solved
  from LB readings (`LB ≈ 0.162·c1 + 0.22·c2`), **not** set to the raw class
  share 0.16, because Class-2 LB value is discounted by pool recall.
- GBM config (`max_depth=6, max_iter=500, lr=0.03, min_samples_leaf=80,
  l2=1.0`) is deliberately regularised: depth 6 admits 3-way channel
  interactions (e.g. lib × agree × lvmax) while leaf-80 + L2 stop it memorising
  individual scaffolds in a rank_train with few thousand molecules.

## 3. Learning-to-rank SOTA beyond pointwise HistGradientBoosting

What the fork does is **pointwise classification with listwise features**:
each candidate row is scored independently, but rank/z/gap/agreement columns
smuggle the query context in. This is close to McRank territory and works, but
it optimises log-loss per row, not MRR per query. Options, ordered by
expected value under offline-CPU feasibility:

1. **LightGBM LambdaMART (`objective: lambdarank`, MRR/NDCG gains).**
   Still GBDT-based SOTA for tabular ranking (ICML'23 Lyzhin et al.: LambdaMART
   vs YetiRank vs StochasticRank — LambdaMART-family wins on most tabular
   benchmarks; Yandex production rankers use YetiRank, a LambdaMART variant).
   Lambda gradients weight pairwise swaps by the **metric delta** (1/rank
   movement), so training effort concentrates on top-of-list swaps — exactly
   MRR@25. Drop-in: same 31 features + `group` = nc per query, label = Y.
   Expected **+0.003–0.008 LB**: the headroom is precisely the rows where
   log-loss cares about p=0.02-vs-0.05 distinctions deep in the list that MRR
   never rewards. Risk: needs the LightGBM wheel offline; validate grouped CV
   before trusting.
2. **CatBoost YetiRank (direct MRR optimisation).** YetiRank with an MRR-style
   loss optimises a smoothed version of the actual metric rather than an NDCG
   proxy; the Lyzhin et al. "improved YetiRank" variant is the current SOTA on
   several LETOR benchmarks. Same data format as LambdaMART. Worth trying
   alongside LightGBM — whichever wins grouped CV ships; blending both is
   usually +small. Same cost class.
3. **Two-tower / two-ranker blend (megayak pattern).** Train one ranker on
   Class-1-like rows, one on Class-2-like rows (or seen-library vs held-out
   split), blend by a query-level gate (`lvmax`, `agree`). This is a coarser,
   more sample-efficient version of what W1-priors approximate, and it fixes
   the "public FP weights saw most libraries" over-trust leak (0.49 held-out
   vs 0.76–0.82 seen). Cheap: 2× HGB fits on row subsets.
4. **ListNet / ListMLE / deep listwise (skip for now).** Softmax-cross-entropy
   over the candidate list (ListNet) or Plackett-Luce likelihood of the true
   permutation (ListMLE) are the right losses *in theory*, and listwise
   transformers (RankFormer, KDD'23) extend them with cross-candidate
   attention. In practice they need far more queries than ~10³ molecules to
   beat GBDTs on 31 dense features, and per-query score translation-invariance
   (RankFormer §3: pair/listwise losses ignore absolute scale) throws away the
   cross-query calibration the agreement block currently exploits. Revisit only
   after rank_train grows 10× (e.g. megayak's 2250-mol × 7-library rows).
5. **Calibrated-vs-sharp.** Only **within-query** order matters here
   (per-molecule argsort → top 25), so sharp listwise scores are safe at
   inference. But `rank_proba`'s **mean over 8 models** and any future
   cross-model blending need comparable scales — keep the pointwise HGB (or
   Platt/temperature-scale the LambdaMART outputs) as the calibrated combiner
   on top of sharper base rankers. Stack, don't replace: LambdaMART scores as
   2–3 extra features into the existing HGB ensemble is the lowest-risk
   upgrade path.

Feature directions that would feed any of the above (all CPU-computable):
score margins `ap − second-best ap`, `raw gap to runner-up`; count of analogs
above sim thresholds; `nrm`-side argmax flag (asymmetric with col 20);
per-channel argmax coincidence flags (lib==analog, analog==model top-1);
`log(nc)` interactions precomputed (`lvmax·log nc`); adduct-shifted library
max (megayak +0.01–0.02 Class-1); same-polarity analog restriction flag.

## 4. Seed-bagging and the ±0.006 noise floor

Two measured noise figures ship with the fork: **±0.006 LB seed noise**
(`CFG.SEEDS` comment) and 0.0072 local-CV seed noise (upstream cells 24–25).
Sources: HGB histogram subsampling/tie-breaking across `random_state`, and the
tiny effective sample (hundreds of test molecules; each molecule contributes
1/rank quanta ≈ 0.04–1.0 to its MRR term). The fork's answer: **4 seeds × 2
priors = 8 models, mean proba** (`rank_proba`). Bagging shrinks fit variance
≈ 1/√8 while the prior-dim hedges regime bias — variance reduction and bias
hedging in one average.

Statistics for 5 submissions/day:

- Treat ±0.006 as 1σ of the *submission* noise (seed + LB-sample noise
  convolved). A single-submission delta < 0.007 is indistinguishable from
  noise — do not ship on it.
- Require **|ΔLB| ≥ 0.012 (2σ)** on one sub, or **same-sign Δ ≥ +0.006 on two
  consecutive subs**, before promoting an idea to the trunk. The fork's own
  +0.003 prior-hedge gain would *not* pass the single-sub bar — it passed by
  replication across seeds/CV, which is the point.
- Spend the 5 subs as: 1 = current-best re-sub (anchor; detects LB reshuffles),
  2–3 = one change each, 1 = spare/rollback. **One change per sub** or deltas
  are uninterpretable; the anchor sub is what separates "idea worked" from
  "board moved".
- Offline gate first: grouped-by-molecule CV (never grouped-by-spectrum —
  spectra of one molecule share the answer), with source-library masking for
  Class-1 (mask the whole library, else identical-spectrum retrieval fakes
  1.000) and held-out-structure masking for Class-2. Ship only what scored
  offline *and* clears the LB bar above. Fix seeds everywhere except the
  deliberate 4-seed bag.
- Track per-class MRR separately (Class-1 ≈ saturated; all movement is
  Class-2). A +0.01 LB with −0.05 Class-1 / +0.08 Class-2 is a different fact
  than uniform +0.01 and suggests different follow-ups.

## 5. Concrete upgrades, ranked by expected LB delta (offline CPU feasible)

| # | Upgrade | Why | Expected ΔLB | Cost |
|---|---------|-----|--------------|------|
| 1 | LightGBM LambdaMART on same 31 feats (+ `group`), blend/stacked with HGB | Optimises top-of-list swaps, not per-row log-loss; §3.1 | +0.003–0.008 | CPU; needs wheel |
| 2 | Two-ranker Class-1/Class-2 blend gated by `lvmax`/`agree` | Removes regime compromise now averaged over; fixes seen-library over-trust | +0.002–0.006 | 2× HGB fit |
| 3 | Margin/count features (runner-up gaps, #analogs > thresholds, argmax-coincidence flags) | Top-of-list decisions live in gaps, current feats under-describe them | +0.002–0.005 | CPU feature code |
| 4 | Adduct-shifted lib max + same-polarity analog flag (megayak deltas) | Same molecule as different adduct keeps neutral losses; measured +0.01–0.02 Class-1 | +0.001–0.004 | CPU (reuse kernels) |
| 5 | W1 grid {0.3,0.42,0.5,0.6} × seed bag 8, LB-solved weights | Fork's 2-prior hedge is coarse; finer prior sweep + more seeds | +0.001–0.003 | 4× HGB fits |
| 6 | Platt/temperature calibration of stacked scores | Keeps cross-model averages comparable (§3.5); protects agreement logic | +0.000–0.002 | trivial |
| 7 | CatBoost YetiRank-MRR as second listwise view | Independent loss family; blends well with LambdaMART | +0.000–0.003 | CPU; needs wheel |
| 8 | Deep listwise (ListNet/ListMLE/transformer) | Data-starved at ~10³ queries; revisit after 10× rank rows | −0.005–+0.002 now | GPU + data pipeline |

Suggested order: 3 → 1 → 2 → 5 → 4 → 6 → 7. Features first (they help every
ranker), then the loss upgrade, then regime/blend refinements. Skip 8 until
rank_train is rebuilt at megayak scale (2250 molecules × 7 libraries).
