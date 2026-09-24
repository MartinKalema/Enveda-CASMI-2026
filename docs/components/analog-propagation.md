# Analog Propagation Channel (Channel 2) — Components Guide

Baseline: fork of top solution (LB 0.328), full code in `kernels/fork-034/notebook.ipynb` (committed).
Cell map (0-indexed `cells[]`): C0 header/mermaid, C1 `CFG`, C2 I/O bootstrap,
C3 numba kernels (`_clean`, `entropy_sim`, `entropy_sim_shift`, `search`, `search_shift`),
C4 adduct physics + `load_library`/`lib_window`/`build_rep`, C5 candidate pool,
C6 `lib_sim`/`analog_sim`/MetFrag-lite, C7 FPNet, C8 `rank_features`+GBDT ensemble,
C9 inference pipeline, C10 diagnostics.

## 1. How it works (with cell references)

**Rep set — `build_rep` (C4, last fn).** Collapses 2.54M library spectra to one
representative per `inchikey14`: the spectrum with max peak count (`np.diff(off)`).
Sorts reps by neutral mass (`rep_nm`) for binary-search windowing. Output
`rep, rep_key, rep_nm, rep_ad` (~hundreds of k scaffolds). Rationale: richest
spectrum = most fragment information per structure; dedup prevents over-counting
structures with many replicate spectra dominating top-K.

**Analog retrieval — `analog_sim` (C6, second fn).**
Per query molecule with median neutral mass `target` (C9: median over spectra's
`neutral_mass()`):
1. Window reps to `target ± CFG.ANALOG_WIN` (±200 Da) via `searchsorted` on `rep_nm`.
2. Same-adduct prefilter: `m = wad == ad` per query spectrum; fallback to all
   in window if zero same-adduct refs.
3. `shift = target - rep_nm[window]` per candidate (neutral-mass delta, float32).
4. `search_shift` (C3): cleans both spectra (`_clean`: 0.2% floor, top-256,
   L1-norm, entropy reweight if S<3), then `entropy_sim_shift` = max(direct
   entropy sim, shifted entropy sim with library m/z + shift). Numba `prange`
   parallel over candidates.
5. Max-pool over query spectra per structure (`agg[k] = max s`), sort desc,
   keep top `CFG.N_ANALOG = 100`.

**Scoring into ranker — `rank_features` (C8).** For each mass-window candidate
(median 56, ±8.5 ppm, C9 `pool.window`), compute Tanimoto between candidate FP
and each analog FP (`cf @ af.T` corrected), weighted by analog sim:
- `ap = max_j tan_ij * sim_j^p`, p = `SIM_POWER = 4` (the headline feature);
- `a1 = max_j tan_ij * sim_j` (p=1 variant); `best_tan`, `top_tan`, `mean_tan`
  (sim^p-weighted mean), `top_sim`. Only analogs whose key exists in pool
  (`pool.k2i`) contribute (`ids/sims` filter, C9). CFG comment records tuning:
  p=1 → 0.498, p=3 → 0.521, p=4 → 0.525 (Class-2 MRR). So the channel emits
  ~9 of the 31 ranker features; GBDT ensemble (8 models: 2 W1 priors × 4 seeds)
  learns when to trust `ap` vs library sim vs FPNet logit.

**Cost.** Window ±200 Da × ~300k reps → few thousand candidates per query
spectrum × numba entropy sim; top-100 kept. Feasible in 9h CPU (the LB 0.328
kernel runs it for all test molecules).

## 2. Chemistry assumption + GNPS evidence

**Assumption:** a congener/analog differs from a library scaffold by a single
localized modification of mass dM (|dM| ≤ 200 Da); unmodified substructure
fragments appear at identical m/z, modified-substructure fragments appear
shifted by exactly dM. Then `max(direct, shifted)` entropy similarity is high
and the analog's fingerprint is a good proxy vote for the query's fingerprint
(`sim^p · Tanimoto` propagation).

**When true:** single inert R-group changes preserving fragmentation mechanism —
glycosylation/deglycosylation (±162.05), methylation/demethylation (±14.02),
hydroxylation (±15.99), halogenation (Cl ±33.96/34.97, Br ±77.91),
prenylation (+68.06), acetylation (+42.01), Lys→Ser type residue swaps (−41 Da
stenothricin case). Natural-product series (lipopeptides, flavonoid glycosides,
fentanyl analogs) are the textbook wins.
**When false:** (a) multiple distributed modifications (score decays sharply —
MS2Query paper notes modified cosine collapses under ≥2 edits);
(b) modification at the charge-retention/fragmentation-directing site reroutes
fragmentation (no shared peaks in either view); (c) small molecules (<200 Da,
few peaks) where 6-peak GNPS thresholds fail; (d) scaffold hops with same mass
but different connectivity (high Tanimoto required but sim low, or vice versa);
(e) cross-mode comparisons (+ vs −) where fragmentation chemistry differs.

**GNPS molecular-networking evidence (Dorrestein/Bandeira lineage):**
- Watrous et al., PNAS 2012 (living colonies): modified-cosine alignment —
  peaks match if `|mz(p)−mz(p′)|≤t` OR `|mz(p)+ΔM−mz(p′)|≤t`, bipartite max-weight
  (cosine) assignment; edges kept if cos ≥0.5–0.7, ≥6 matched peaks, top-10
  mutual neighbors, ΔM ≤ 400 Da / 45% mass. This is the direct ancestor of our
  `entropy_sim_shift`.
- Yang et al., J Nat Prod 2013 (dereplication): 58 molecules incl. analogs
  dereplicated across polyketides/alkaloids/peptides; analog capture is the
  stated advantage over exact-match dereplication.
- Wang et al., Nat Biotechnol 2016 (GNPS): 35.2% of unidentified nodes connect
  to a related spectrum at cosine ≥0.8 (44.7% at ≥0.65); stenothricin-GNPS
  discovery (−41 Da, Lys→Ser, NMR/Marfey/genome validated) proves single-dM
  propagation finds genuinely novel congeners. Only ~1.9% exact +1.9% analog-identified:
  most chemical space is analog-only — exactly our competition regime.
- Bittremieux et al., JASMS 2022 (955k peptide + 10M small-molecule pairs):
  modified cosine > neutral-loss-only > cosine in all cases; performance depends
  on modification position/type and compound class — constant-dM is best on
  average but not uniformly.
- Stein/Moorthy HSS (NIST hybrid search, Anal Chem 2017): same constant-dM
  idea as `hMF(Q,L,Δm)` for EI spectra, incl. designer-drug nearest neighbors;
  requires query molecular mass (we supply it via adduct-derived neutral mass).

## 3. Competition constraints it exploits

1. **Novel NPs with no reference spectra (Class-2).** Exact `lib_sim` (C6, ±8.5 ppm
   window) returns ~0 for unseen structures; analog channel is the only spectral
   evidence that fires (baseline: V16 analog addition lifted Class-2 MRR
   ~0.17→0.55, V17 →0.61; LB 0.205→0.266).
2. **Large candidate ambiguity at fixed mass.** Median 56 isobaric candidates in
   ±8.5 ppm window (C10 diagnostic); `ap` breaks ties by scaffold votes from
   outside the mass window.
3. **Multi-spectrum molecules.** Max-pool over adducts/CIDs in `analog_sim`
   mirrors multi-spectrum consensus (V15 lesson).
4. **COCONUT-heavy prior.** Candidates are natural products; NP families are
   congener-rich, so the GNPS single-modification prior is well matched.
5. **No test-label leakage risk.** Reps come only from `train.parquet`; test
   structures are novel by design, so only analog (not exact) transfer is legitimate.

## 4. Deep research: better retrieval and scoring

**Variable-shift / beyond-constant-dM alignment.**
- GNPS modified cosine allows ONE global shift; neutral-loss matching (subtract
  precursor) is shift-free but weaker alone (Bittremieux 2022: loses to modified
  cosine). MZmine analog search now offers three modes: modified cosine,
  cosine-no-precursor (pure fragment overlap, good for pseudo-spectra), MS2DeepScore.
- Flash-Entropy (Nat Methods 2024, 6B-spectra index): formalizes identity / open
  (fragment-only) / neutral-loss / hybrid searches with entropy similarity;
  hybrid = each query ion matches fragment OR neutral-loss table once, fragment
  priority. Our `entropy_sim_shift` is hybrid-lite (max of two views, not
  per-peak OR). Full per-peak hybrid assignment is the principled upgrade.
- Multi-shift reality: two-edit congeners need two shifts; practical approximations:
  (a) score = max over {0, dM} per-peak assignment (Hungarian, matchms
  `ModifiedCosineHungarian`) instead of max over two whole-spectrum scores —
  strictly more expressive at same window; (b) top-2 dM hypotheses from
  common-loss list; (c) fragment-only + loss-only two-channel features and let
  GBDT combine them (cheap, no assignment change).

**Spec2Vec (Huber et al., PLoS Comp Biol 2021).** Word2Vec over co-occurring
fragments/losses; fixed-length embeddings → cosine in embedding space. Trained
on ~95k GNPS spectra (~30–40 min i7 CPU). Findings: correlates with Tanimoto
better than (modified) cosine; 0.14 s/query vs 4–8 h for brute-force cosine on
78k library (their benchmark); top-10 contains Tanimoto>0.6 analog for 60% of
novel queries (random baseline 1%), mean best Tanimoto >0.8 for >400 Da queries.
Use case for us: **retrieval prefilter** (fast top-K over full rep set, no
±200 Da cutoff) followed by expensive entropy-hybrid rescoring — recall booster,
not ranker replacement.

**MS2Query (Nat Commun 2023).** Combines Spec2Vec + MS2DeepScore + precursor-mass
features in a random-forest analog-quality classifier. Analog benchmark (no exact
match in library): MS2Query mean Tanimoto 0.63 at 35% recall vs 0.45 for modified
cosine at same recall; 80 spectra/min laptop vs 10.6/min modified cosine @100 Da
window. Lesson: **learned combination of scores beats any single p/threshold** —
supports adding Spec2Vec/MS2DeepScore columns as ranker features rather than
replacing `ap`.

**MS2DeepScore (Huber 2021+).** Supervised Siamese embedding trained to predict
Tanimoto directly; best single-score analog retriever in MS2Query ablations.
Heavier (needs train split discipline: train only on train.parquet spectra to
avoid leakage) but CPU-feasible for top-K rescoring.

**p/weighting schemes.** Current `sim^p·tan, p=4` is a hard-AND gate: kills all
but near-identical analogs (0.5^4=0.06). Literature + tuning notes suggest:
- p is a precision/recall knob; p=3→4 gave only +0.004 — saturated. Better:
  **two-channel features** (`p=1` recall + `p=4` precision already partially
  present as `a1`/`ap`; add `p=2` and rank-based reciprocal weights).
- Replace `max` with **top-k soft vote** (mean of top-3 `sim^p·tan`): robust to
  single false-positive analog; GNPS top-10 mutual-neighbor logic agrees.
- Calibrate sim first: entropy sim is uncalibrated across entropy levels
  (Flash-Entropy: query time/accuracy varies S=1–4); per-query z-score or
  `sim − median(window sims)` before powering.
- Weight by **mass-shift prior**: common biotransformations (Δ list:
  ±14.02, ±15.99, ±18.01, ±28.03, ±42.01, ±68.06, ±162.05…) up-weighted; rare
  dM down-weighted. GNPS mass-defect propagation logic.
- Multiply by **matched-peak count factor** (GNPS ≥6 rule): `sim * min(1, n_match/6)`
  suppresses few-peak chance alignments that power-weighting alone keeps when
  sim is spuriously high.

**Analog count selection (80→100 history).** No theory favors 100; it's a
compute cap. Better: **adaptive K** — keep sim ≥ max(0.3, median+2σ) up to cap
200; always keep ≥5 if any. Ranker already has `mean_tan`/`top_sim` so larger K
mostly denoises via mean features. Offline test: sweep K ∈ {25,50,100,200} ×
window ∈ {100,200,400} on local Class-2 split; expect plateau ~100–200 with
long-tail recall from 200–400 Da (GNPS allowed 400 Da).

## 5. Concrete upgrades (ranked by expected LB delta, 9h CPU feasible)

1. **Per-peak hybrid (Hungarian) rescoring of top-100 analogs** (+0.004–0.010).
   Replace whole-spectrum `max(direct, shifted)` with one-to-one max-weight
   assignment over union of direct+shifted edges (matchms Hungarian logic,
   numba). Fixes double-counting a query peak in both views; biggest win on
   partial-overlap congeners. Cost: 100 pairs × small assignment per molecule —
   trivial.
2. **Common-dM prior weight + matched-peak gate as new ranker features**
   (+0.003–0.008). Add `ap_common` (bonus if dM within 10 mDa of known-loss
   list), `n_match` count, `sim·min(1,n/6)`. Retrain GBDT (minutes). Zero
   inference risk.
3. **Adaptive-K + ±400 Da window with same numba kernels** (+0.002–0.006).
   Recall-only change; ranker learns to ignore added noise via `top_sim`/
   `mean_tan`. ~2× retrieval cost, still inside 9h (window 400 Da ≈ doubles
   reps in range; numba prange absorbs it).
4. **Spec2Vec retrieval arm (offline-trained on train.parquet only)** (+0.002–0.006,
   mostly Class-2 recall on >400 Da). Train gensim Word2Vec ~40 min CPU; index
   rep embeddings; per query take union(Spec2Vec top-50, entropy top-100) →
   existing `ap` features computed on union + one `spec2vec_max` feature.
   Validates Huber's >400 Da finding directly.
5. **Top-3 soft vote + p=2 channel** (+0.001–0.004). Two extra columns in C8,
   no retrieval change. Cheap ablation; guards against single-analog false positives
   that p=4 max amplifies.
6. **Neutral-loss-only channel feature** (+0.001–0.003). Score query neutral
   losses vs analog neutral losses (precursor − fragment); catches charge-
   retained series where fragments shift but losses don't. ~1 extra search_shift
   pass on transformed spectra.
7. **MS2DeepScore rescoring** (+0.002–0.005 but risky). Needs leakage-safe
   training + threshold calibration; do only after 1–4 land. Skip if time-boxed.

## 6. Same-adduct restriction scored 0.323 (neutral) — interpretation

The tested variant forced the analog channel to same-adduct refs only (removing
the `ones(...)` recall fallback in `analog_sim`, C6). LB 0.323 ≈ baseline:
**no gain, no loss.** Implications:
- **Analog evidence is substantially adduct-transferable.** After neutral-mass
  normalization (`shift = target − rep_nm`), scaffold fragments coincide across
  adducts within a polarity; the entropy matcher already compares fragment
  patterns, not precursors. Cross-adduct analogs therefore contribute true
  scaffold votes (Tanimoto signal), not just noise.
- **Fallback rarely fires but rescues rare adducts.** Most queries ([M+H]+,
  [M−H]−, [M+Na]+) have same-adduct refs; restriction changes nothing for them.
  For rare adducts (2M/3M multimers, metal adducts) the fallback is the only
  source of analogs — removing it loses those molecules, offsetting any
  precision gain elsewhere. Net neutral.
- **Fragmentation chemistry is polarity- not adduct-specific.** Same-polarity
  different-adduct spectra (e.g. [M+H]+ vs [M+Na]+) share neutral losses and
  backbone cleavages; the failure mode the restriction targeted (adduct-specific
  fragments misleading `sim`) is empirically small at p=4 (power gate already
  suppresses weak cross-adduct hits).
- **Decision:** keep same-adduct preference + fallback (current code). Better
  upgrade: **soft adduct penalty feature** (`same_adduct_frac`, `best_cross_adduct_sim`)
  instead of hard restriction — lets GBDT learn per-class adduct transferability
  rather than imposing a global rule the data just rejected.
