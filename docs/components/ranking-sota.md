# Ranking SOTA for tiny-query retrieval: papers + new designs

Context: Kaggle MS/MS retrieval, MRR@25, ~400 test queries, binary labels
(one true structure per query, ties everywhere else), 31-feature seed-bagged
pointwise HGB at LB 0.155. Read `ranker-calibration.md` first — it covers the
fork's GBM, agreement features, W1 priors, seed-bagging. This doc goes beyond
it: 12 papers downloaded to `docs/papers/ranking/` (all `file`-verified PDFs,
text in `docs/papers/ranking/txt/`), per-paper verdicts for OUR regime, then
3 new ranking schemes ranked by expected LB delta + offline-CPU feasibility.

Regime facts that filter every verdict below: (a) ~10²–10³ training queries,
not 10⁴–10⁵ — every LTR paper here trains on 6k–20k queries (Lyzhin §4) or
19k-query Yahoo/MSLR sets (PLRank §4); (b) binary relevance with massive
ties (1 positive vs up to hundreds of negatives per query); (c) labels noisy
(Class-2 wrong-isomer library hits score 0.5–0.8); (d) only within-query order
matters (MRR@25), but cross-model score averaging needs comparable scales;
(e) submission noise ±0.006 — nothing below +0.007 single-sub ships.

## 1. Per-paper verdicts (what each paper says for ~400 queries, noisy labels)

### 1.1 Burges 2010, "From RankNet to LambdaRank to LambdaMART" (MSR-TR-2010-82)
Mechanics that matter: λ-gradients = RankNet pairwise gradient × |Δmetric|
from swapping the pair (§4.1, eq.6) — training effort concentrates on
top-of-list swaps, exactly MRR@25. MART Newton leaf step scales as 1/σ so
σ choice is irrelevant (§7) — one fewer hyperparameter to tune on tiny data.
LambdaMART updates splits/leaves on ALL queries at once, unlike LambdaRank's
per-query weight steps (§7) — better gradient averaging when queries are few.
Sleeper: §7.1 "How to Optimally Combine Rankers" — sweep a scalar blend
α of two rankers' scores, metric changes only finitely often, keep the best α;
or equivalently train a linear LambdaRank on base-model scores as features.
That is a license for our stacking designs (§3): learned combination, not
hand weights.
Verdict: ADOPT the λ×|ΔMRR| idea; ADOPT §7.1 blend-sweep as the combiner
protocol. Skip pure-LightGBM-LambdaMART default — see §1.3 for why YetiLoss
is the better GBDT-listwise pick here.

### 1.2 Gulin et al. 2011, "Winning the Transfer Learning Track … with YetiRank"
Two ideas that map 1:1 onto our failure modes. (1) Label-noise model:
pair weight w_ij = N_ij · Σ_u,v 1[u>v] p(u|l_i)p(v|l_j) (§3.2, eq.7–8) with an
explicit editor confusion matrix — our Class-2 rows ARE label-confused
(wrong isomers with high library sims are "editors" disagreeing). They even
infer the confusion matrix from near-duplicate feature buckets when double
labels are unavailable (§4.2) — we can approximate it from same-mass
isomer score distributions. (2) Stochastic neighbor-pair weights: perturb
scores with logistic noise, re-rank 100×, increment weight 1/R only for pairs
that land adjacent (§3.2) — smoothing + top-focus in one step; plus a direct
least-squares pairwise leaf solve instead of LambdaRank's pointwise
reduction (§3.4, cost O(N_q·N_d²) + leaf-matrix inverse — fine at our scale).
Headline result: won the transfer track WITHOUT using the larger
transfer-from set (§5) — i.e. YetiRank wins exactly when auxiliary data is
distributionally suspect, which is our "public FP weights saw most libraries"
situation.
Verdict: ADOPT noise-weighted pairs + neighbor smoothing. The confusion
matrix gives a principled replacement for the W1 scalar hack.

### 1.3 Lyzhin et al. 2023 (ICML), "Which Tricks Are Important for Learning to Rank?"
The paper that most directly answers "LambdaMART vs YetiRank vs pointwise
on small data". Six datasets, CatBoost-unified implementations, tuned +
default-hparam comparisons. Findings ranked by relevance to us:
1. YetiRank ≥ YetiLoss > StochasticRank > LambdaMART on most (dataset,
metric) cells; YetiLoss is SOTA for MRR/MAP (Table 2). For OUR metric
(MRR) the MRR-targeted YetiLoss is the single best-documented choice.
2. Smoothing is ESSENTIAL: no-smoothing loses ~2 NDCG points everywhere
(Table 3; NDCG@10 50.76→48.49 Web10K, MAP 62.62→61.32). Shape
(Logistic vs Gaussian) barely matters. Tiny data ⇒ keep CatBoost defaults
(10 permutations) or more, never zero.
3. Neighbor-pair restriction (|p_i−p_j|=1) vs all-pairs: no stable significant
difference; k=2 is a safe efficient pick (Fig.1, Tables 5–10). Don't pay
LambdaMART's all-pairs cost.
4. QueryRMSE (plain per-query-averaged RMSE — barely "ranking" at all) is
competitive with LTR methods on several cells, even best on Yahoo S2
NDCG@10 and Yahoo S1 ERR (Table 2). On tiny noisy data a pointwise
regression baseline is not a strawman — it anchors every stack.
5. CatBoost oblivious trees generalize better (smaller train–test gaps than
LightGBM almost everywhere, Tables 23–26) — tree-structure choice matters
more than loss choice at small N.
6. Convex surrogate (YetiLoss) usually beats direct non-convex optimization
(StochasticRank) — skip StochasticRank/SGLB here: heavier, needs tuning
(diffusion-temp, mu), no win on small data.
Caveat: their smallest set (Yahoo S2, 1266 train queries) is still 3× ours —
shrink all effect sizes and widen the tuning budget toward regularization
(depth 6, l2-leaf-reg high, lr low).
Verdict: PRIMARY listwise upgrade = CatBoost YetiLoss targeting MRR
(+ QueryRMSE anchor). Prefer CatBoost over LightGBM (ordered boosting +
oblivious trees, see §1.11). Do NOT ship LambdaMART alone.

### 1.4 Xia et al. 2019, "Plackett-Luce Model for Learning-to-Rank" (PLRank/ListMLE)
First single listwise model to match/beat LambdaMART on Yahoo-2010 +
MSLR-30K (Table 1: PLRank NDCG@10 0.7902 vs LambdaMART 0.7809 Yahoo).
Three warnings that bite at our scale: (1) LINEAR ListMLE is unstable and
needs #features ≫ #docs/query to be consistent (Fig.2: beats coordinate
ascent only past ~100–200 features); we have 31 features and ~10² docs/query
— linear ListMLE is contraindicated, GBDT-PL only. (2) Overfitting "often
occurs in small data sets, while in large datasets log-likelihood correlates
with ranking measures very well" (§3.3) — direct quote against using PL loss
as our only objective at ~400 queries. (3) Ties need multiple ground-truth
permutations (obj=3 recommended, §3.2.2/§4.3) — our binary labels are ALL
ties below rank 1, so obj≥3 sampling is mandatory, not optional.
Verdict: GBDT-PLRank as a DIVERSITY view in a blend (different loss family
from YetiLoss), never standalone. Practical recipe: K=10 (MRR@25-adjacent),
obj=3, same cost class as LambdaMART. Expected marginal over YetiLoss
small; value is decorrelation for the stack.

### 1.5 Luo et al. 2015, "Stochastic Top-k ListNet"
ListNet Top-1 softmax is "a harsh approximation" that discards partial-rank
info (§1); sampling small permutation subsets + Top-2/3 with adaptive
(label- or model-score-proportional) sampling beats full ListNet while being
~1000× faster (Table 1: conventional Top-2 2275s vs stochastic ~2–3s).
Two caps for us: gains saturate past k=2–3 and destabilize past k=4 (Fig.5);
and with coarse 0/1/2 labels, Top-k advantage shrinks because most sampled
lists are all-tied (§5) — our binary labels are the extreme of that regime,
so expect the Bottom of their gain range. Adaptive sampling ≈ hard-negative
mining: sample lists containing high-scoring negatives.
Verdict: SKIP as a primary ranker (binary ties neuter it). SALVAGE one
idea: adaptive permutation/negative sampling for training ANY listwise
GBDT — oversample queries/lists where a confusing negative outranks the
truth (our Class-2 traps). Cheap, loss-agnostic.

### 1.6 Cormack et al. 2009 (SIGIR), "Reciprocal Rank Fusion"
RRFscore(d) = Σ_r 1/(60+r(d)): no training, no scores, rank-only. Beats the
best individual system, Condorcet, and CombMNZ +4–5% MAP on TREC
(Table 2, p≈0.008–0.04); on LETOR-3 meta-ranking beats every learned LTR
method incl. ListNet/RankSVM/RankBoost (Table 3, all p<0.003). k=60
near-optimal and flat (Table 1) — zero tuning, ideal for 5-subs/day.
Why it matters here: CombMNZ's weakness is EXACTLY our weakness —
"multiplies the sum of the uncalibrated scores … results have higher
variance … some scores are more amenable than others" (§2). Our channels
(entropy sims ∈[0,1], logit dots in nats, explained fractions) are maximally
"unamenable" to joint score arithmetic — rank-only fusion sidesteps the
calibration problem the 31-feature GBM solves by learning. Diversity note:
RRF wins "because it harnesses diversity … one or two systems ranking a
document highly can substantially improve it" (§2) — our channels ARE
diverse (frag⊥analog corr ≈0.058).
Verdict: ADOPT RRF(k=60) over per-channel ranks as the zero-shot fusion
baseline every learned combiner must beat offline. Do NOT use CombSUM/MNZ
on raw scores; if score-fusion is wanted, fuse only post-Platt calibrated
probabilities (see §1.7).

### 1.7 Niculescu-Mizil & Caruana 2005 (ICML), "Predicting Good Probabilities"
The calibration paper with a learning-curve answer for tiny data. (1)
Boosted trees/decision-tree ensembles show sigmoid distortion (mass pushed
away from 0/1) — Platt scaling is the matched fix; naive Bayes needs
isotonic instead (inverted-sigmoid distortion). Our HGB posteriors are
boosted-tree outputs ⇒ Platt/temperature, not isotonic. (2) Bagged trees
are already well calibrated; bagging a high-variance model then calibrating
is belt-and-braces — endorses the fork's 4-seed bag AND calibrating after.
(3) THE small-data rule: Platt beats isotonic for calibration sets <200–1000
points; isotonic only wins at ≥1000 (Fig.7, §5). Our calibration slice
(hundreds of molecules) is firmly Platt territory. (4) Calibration on small
sets HURTS already-calibrated models (NN/bagged trees, §5) — so calibrate
only the sharp listwise scores, never the HGB average itself. (5) After
calibration: boosted trees + RF + SVM best; calibration can't rescue weak
base models — stack strong rankers only.
Verdict: Platt (2-param, or 1-param temperature) on held-out molecules for
every sharp base score before blending; isotonic FORBIDDEN below ~1000
calibration rows. This hardens ranker-calibration.md §3.5/"upgrade #6" with
numbers.

### 1.8 Guo et al. 2017 (ICML), "On Calibration of Modern Neural Networks"
Temperature scaling (single-parameter Platt) beats vector/matrix Platt
variants on vision tasks (Table 1) — "miscalibration is intrinsically low
dimensional". Two transfers: (a) our FPNet-logit channel likely needs only
1-param rescaling per spectrum, not a learned map — fewer params, less
overfit at N≈400; (b) NLL keeps improving after accuracy saturates, i.e.
NLL-overfit without accuracy-overfit (Fig.3, §3) — monitor ECE (eq.3), not
log-loss, when tuning any calibrated combiner; log-loss will happily
over-sharpen. MCE (eq.5) for the high-risk Class-1/Class-2 gate.
Verdict: 1-param temperature for neural-channel rescaling; ECE+binned
reliability plots as the combiner-selection metric alongside MRR.

### 1.9 Vovk & Petej 2014, "Venn–Abers Predictors" (arXiv:1211.0025)
Finite-sample-valid calibration under iid ONLY (no distributional
assumptions). Direct isotonic (DIR) overfits (same data for scoring +
calibration); full Venn–Abers fits 2 isotonic regressions per test point
(one per postulated label); Simplified VA (Algorithm 2) reuses one scoring
function — O(l log l) precompute + O(log l) per test point, CPU-trivial.
Never suffers infinite log-loss (Lemma 2). Empirical: VA/SVA fix the ∞
log-loss of raw Weka scores across classifiers/datasets (Table 1).
The (p0,p1) interval width is epistemic uncertainty FOR FREE — wide on
small samples.
Verdict: Simplified Venn–Abers as the abstention/cascade signal (§3, design
C): width of (p0,p1) on the Class-1-vs-2 gate tells us when to trust library
evidence. Valid guarantees need exchangeable calibration molecules —
group by molecule, never by spectrum.

### 1.10 van der Laan et al. 2025, "Generalized Venn and Venn–Abers Calibration"
2025 update that answers the small-sample objection: Venn sets WIDEN on
small calibration sets (quantifying epistemic uncertainty) and shrink to the
point prediction asymptotically; each contained prediction is marginally
calibrated in finite samples. Plus Venn MULTICALIBRATION: finite-sample
calibration across subpopulations — read: calibrate Class-1 and Class-2
queries separately with guarantees, replacing the W1 scalar with per-regime
calibrated maps.
Verdict: per-class (Class-1/Class-2) Venn/Platt calibration maps as the
final combiner layer (design B/C). The set width doubles as a
"don't-trust-library" detector.

### 1.11 Prokhorenkova et al. 2018, "CatBoost: unbiased boosting…" (arXiv:1706.09516)
Prediction shift = target leakage in ALL classic GBDTs: F(x_k) on train has
a different distribution than F(x) on test because gradients saw the targets;
ordered boosting (permutation-driven residuals) fixes it. Measured: Ordered
mode's edge is LARGEST ON SMALL DATASETS (<40K rows; Adult/Internet,
Fig.2, §6) and Plain degrades as data is filtered down. Oblivious trees
(same split per level) resist overfitting + fast inference. Our rank_train
(10³ molecules × 10² candidates) is two orders below their "small" — the
regime where Ordered helps most.
Verdict: CatBoost-Ordered is the mandated GBDT implementation for every
design below (ranker, gate, combiner). This is the strongest small-data
argument in the whole set — and it compounds with §1.3 finding #5.

### 1.12 Buyl et al. 2023 (KDD), "RankFormer: Listwise Learning-to-Rank…"
Negative result that saves us a GPU detour + two salvaged ideas. (1) Even
with 10k-tree budgets, GBDTs beat neural/listwise-transformer rankers on
tabular LTR data (Tab.2 discussion; cites Grinsztajn'22, Qin'21) — confirms
skip-deep-listwise at 31 dense features. (2) Pair/listwise losses are
"completely invariant to global translations of the scores vector … not
necessarily calibrated" (§4, citing Yan et al.) — independent corroboration
of keeping a pointwise calibrated combiner on top of sharp listwise bases.
(3) LISTWIDE signal: absolute list quality (their [CLS] head) helps when
relative feedback is uninformative (all-negative lists). Our translation:
every Class-2 query is an "all-weak-evidence list" — a query-level
answerability/Class-1-posterior head is legitimate extra supervision, not a
hack. (4) Distillation RankFormer→GBDT(2000 trees) preserves gains for
production (§7) — any future neural experiment ships via distillation, never
as the inference model.
Verdict: no transformer. Steal the listwide auxiliary head (design B) and
the distill-to-GBDT deployment pattern.

## 2. What this changes vs ranker-calibration.md §3/§5
- Loss pick narrows: YetiLoss-MRR (CatBoost) > YetiRank-ExpDCG >
  LambdaMART for our metric; StochasticRank and deep listwise stay skipped,
  now with measured reasons (Lyzhin Tables 2–3; PLRank §3.3; RankFormer §6).
- Implementation pick: CatBoost-Ordered everywhere (prediction-shift §1.11
  + generalization-gap §1.3), LightGBM demoted to diversity-view only.
- Calibration rule hardened: Platt/temperature ONLY, isotonic forbidden
  below ~1000 cal rows (Niculescu Fig.7); ECE-gated (Guo); per-class maps
  (Venn-multical §1.10) instead of finer W1 grid (their upgrade #5 —
  downgrade it: a grid over one scalar can't express regime-dependent maps).
- Fusion baseline: RRF(k=60) rank-only must be beaten offline before any
  learned combiner ships (§1.6); CombSUM/MNZ on raw scores banned.
- Noise model: confusion-matrix pair weights (§1.2) replace/augment W1;
  adaptive hard-negative sampling (§1.5) for listwise training.

## 3. New ranking schemes (ranked; all offline-CPU, no GPU, no new labels)

### Design A (SHIP FIRST): YetiLoss-MRR + QueryRMSE + RRF, Platt-stacked HGB gate
What: three frozen base views per (query,candidate): (i) CatBoost
YetiLoss-MRR on the 31 features (smoothing on, neighbor k=2, depth 6,
high l2-leaf-reg, ordered boosting); (ii) QueryRMSE pointwise regressor
(same features — the Lyzhin-strong baseline, catches what ranking losses
distort); (iii) RRF(k=60) over the 4 channel ranks (lib/analog/model/frag —
zero-shot diversity view). Platt-scale (i)–(ii) on held-out molecules
(temperature variant for the neural channel first, §1.8). Combiner: the
EXISTING 8-model HGB ensemble retrained on [31 feats + 3 base scores +
agree-gated interactions (base×agree, base×lvmax)] with Burges §7.1
α-sweep for the final linear blend weights, selected by grouped-molecule CV
on MRR + ECE.
Why new: ranker-calibration.md proposes "LambdaMART scores as extra HGB
features"; this replaces the loss (MRR-native YetiLoss), adds the RMSE +
RRF views (pointwise-absolute and rank-only — the two signals listwise
losses provably discard, §1.12), and replaces hand-averaging with an
ECE-gated metric sweep.
Predicted effect: +0.004–0.010 LB (YetiLoss→MRR alignment ≈ Lyzhin
MRR-table margins scaled to our N; RRF view adds Class-2 robustness where
lib noise dominates; stack rarely hurts if CV-gated). Feasibility: 3 extra
CatBoost fits + Platt (seconds) + HGB refit; all CPU; one Kaggle kernel
fits. Risk: YetiLoss-MRR overfits 400 queries → mitigate with smoothing,
seed-bag the YetiLoss (4 seeds) before stacking.
Gate: beats RRF-alone AND current HGB in grouped CV by ≥+0.006 MRR
before it takes a submission slot.

### Design B (SHIP SECOND): two-stage retrieve-then-cross-encode with listwide head
What: Stage-1 (retrieve): cheap pointwise pre-rank — per-channel best
(analog ap, model nrm, lib lv, frag fr) + RRF → top-25 per query (recall
target ≥0.99 of current top-25). Stage-2 (cross-encode): CatBoost
YetiLoss-MRR trained ONLY on top-25 rows with an expanded margin grammar:
runner-up gaps (ap−2nd ap, raw−2nd raw), #analogs above sim thresholds,
argmax-coincidence flags (lib==analog, analog==model-top1), base×log(nc)
interactions — i.e. features that only exist conditional on the shortlist.
Auxiliary listwide head (RankFormer §1.12 steal): same Stage-2 model
multi-tasked (or a sibling GBDT) to predict query-level Class-1 posterior
from query-constant features (lvmax, agree, corr, log nc, top_sim); its
output becomes the calibrated gate that scales the library block
(per-class Venn/Platt maps, §1.10) instead of the W1 scalar.
Why new: the fork scores all candidates with one pointwise model; this
separates recall (cheap, high-breadth) from precision (expensive features,
list-conditional), and adds query-level supervision the current 31 features
approximate only implicitly. Mirrors production retrieve→rerank and the
RankFormer listwise+listwide joint objective, in GBDT form.
Predicted effect: +0.003–0.008 LB, concentrated Class-2 (runner-up margins
decide crowded mass windows; calibrated gate suppresses lone-wolf lib
hits). Feasibility: Stage-1 is existing code; Stage-2 trains on ~25×400 =
10k rows — minutes on CPU. Risk: Stage-1 recall cap — verify ≥0.99 offline
or the ceiling binds; shortlist-distribution shift (train Stage-2 on
Stage-1 outputs, never on full lists).
Gate: Stage-1 recall check first; then same CV bar as A. A and B compose
(B's Stage-2 scores become a 4th view in A's stack).

### Design C (SHIP THIRD / fallback): specialist channels + agreement-gated learned fusion
What: four single-channel specialist rankers (lib-only, analog-only,
model-only, frag-only; small CatBoost-QueryRMSE/pointwise HGBs — each
unconfounded by cross-channel leakage), fused per query by a learned gate:
tiny classifier on (lvmax, agree, agree_a, corr, top_sim, log nc) emitting
channel weights; fused score = Σ w_c·Platt(p_c) with RRF as backoff when
the gate's Venn–Abers interval (§1.9) is wide (high epistemic uncertainty →
trust rank-consensus, not scores). Confusion-matrix pair weights (§1.2)
in specialist training down-weight Class-2 lib-trap pairs.
Why new: inverts the fork (one model, 31 joint features) into mixture-of-
specialists with an explicit, inspectable gating policy; the Venn-width
backoff is a new abstention mechanism for retrieval fusion.
Predicted effect: +0.000–0.005 LB alone (specialists underperform joint
features when channels must interact); value is diagnostic (per-channel CV
isolates which evidence moves) + insurance (RRF backoff bounds worst case).
Feasibility: cheapest of the three (small models, logistic gate). Risk:
gating on 400 queries overfits — strong L2, 2–3 gate features max
(agree, lvmax, corr), or fixed RRF when in doubt.
Gate: ship only the gate+RRF-backoff if it beats pure RRF offline; use
specialists mainly to audit A/B ("which channel moved this query").

Ranking: A > B > C. A is the highest expected delta at lowest risk (all
components have paper-measured wins). B has similar upside with a recall
prerequisite. C is bounded-upside but nearly free and de-risks the others.
Suggested order: RRF baseline (1 day, §1.6) → A (YetiLoss-MRR + stack) →
B Stage-1 recall check + Stage-2 → C gate as audit/backoff. One change per
submission; same ±0.006/2σ promotion rules as ranker-calibration.md §4.

## 4. Offline protocol deltas (beyond ranker-calibration.md §4)
- Grouped-by-molecule CV stays; ADD query-stratified reporting (Class-1 /
  Class-2 MRR + ECE per class) — per-class maps (§1.10) need per-class
  metrics to select.
- ADD RRF-beats-learned check: any combiner must beat RRF(k=60) offline
  (Cormack Table 3 precedent: learned LTR losing to fusion is the norm,
  not an embarrassment).
- Calibration slice discipline: Platt/temperature fit on held-out molecules
  only, never train rows (Niculescu §2.1 bias note); n<1000 ⇒ no isotonic.
- YetiLoss-MRR tuning order: fix smoothing ON → depth 6 → l2-leaf-reg →
  lr → trees (Lyzhin App.A ranges); bag 4 seeds; report train–test gap
  (oblivious-tree gap check, Tables 23–26).
- Adaptive sampling (§1.5): oversample top-25 lists where a negative beats
  the truth for listwise fits; costs nothing, targets Class-2 traps.

## 5. Sources (all in docs/papers/ranking/, text in txt/)
Burges'10 LambdaMART (MSR-TR-2010-82); Gulin'11 YetiRank; Lyzhin'23
Tricks (arXiv:2204.01500); Xia'19 PLRank (arXiv:1909.06722); Luo'15
stoch-ListNet (arXiv:1511.00271); Cormack'09 RRF; Niculescu-Mizil'05
calibration; Guo'17 temperature (PMLR v70); Vovk'14 Venn–Abers
(arXiv:1211.0025); van der Laan'25 generalized Venn (arXiv:2502.05676);
Prokhorenkova'18 CatBoost (arXiv:1706.09516); Buyl'23 RankFormer
(arXiv:2306.05808).
