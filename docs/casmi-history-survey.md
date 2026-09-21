# CASMI history survey: what actually won, and what transfers to Enveda MRR@25

Branch: CASMI-history survey | Date: 2026-09-22 | Scoreboard: v0 0.088 vs top 0.409
Companion doc: `docs/constraints-and-literature.md` (MassSpecGym, MIST/MIST-CF, MS2Mol already covered there — not repeated here).

Sources: CASMI 2012/2013/2016/2022 published results, Blazenovic et al. 2017 shootout
("database boosting is needed to achieve 93%"), DrivenData Mars Spectrometry 1+2 winner write-ups.
CASMI 2014/2017 ran small/low-participation rounds; no separate winning method beyond the ones below.

## Ranked findings (most transferable first)

### 1. Pipeline order is always formula-first, then structure (every winner does this)
SIRIUS team (Duhrkop, CASMI 2022: 94% formula, 26-29% exact 2D structure — best automated entry),
MS-FINDER (Kind, CASMI 2016 Cat 3 winner, 76% challenge wins), CFM-ID (Allen, 2nd by 3 medals).
Formula accuracy ~90%+ coexists with structure accuracy ~25-30%: the formula step is what makes
retrieval tractable (isomer-only candidate sets). Transfer: keep the v1 plan's formula ranker as
stage 1; do not score structures against mixed-formula candidate pools. Obsolete: manual formula
curation (won 2012-2013, Dunn/Nikolic) — no humans in this Kaggle loop.

### 2. Fingerprint-prediction retrieval beats fragmentation simulation head-to-head
CASMI 2016 Cat 2 (in-silico-only, no metadata): 1st IOKR (Brouard, 86 gold), 2nd CSI:FingerID
(Duhrkop, 82 gold), 3rd Vaniya (70 gold). Combinatorial fragmenters (MetFrag, MAGMa+) and CFM-ID
trailed badly in the no-metadata category (~half the Top-1 rate). Transfer: our primary scorer must
be predicted-fingerprint vs candidate-fingerprint (MIST-style, already v1), not MetFrag-style bond
breaking. Fragmentation simulation is a rerank feature, not the ranker.

### 3. Consensus of DIVERSE scorers is the single biggest documented lift
Blazenovic 2017 post-contest: best single tool Top-1 ~50-60%; voting/consensus of MetFrag + CFM-ID
(orthogonal principles: combinatorial vs probabilistic-generative) jumped well above either alone;
adding MS-FINDER/MAGMa on top added almost nothing (correlated errors). Transfer: build exactly two
diverse scorers (fingerprint-neural + CFM-style/in-silico-fragmentation or neutral-loss rules) and
fuse by voting/consensus or a LightGBM reranker. Do NOT stack four similar scorers — waste of the
9h budget. This is the cheapest expected LB gain after the formula step.

### 4. Metadata/database boosting dominates whenever rules allow it — check the rules
Same shootout: in-silico-only consensus ~50-60% Top-1; adding database presence + MS/MS library
match ("DB + MS/MS boosting") hit ~93% Top-1 / ~98% Top-10. CASMI 2016 Cat 3 (metadata allowed)
Top-1 ~76% vs Cat 2 (no metadata) ~30-40%: the gap is almost entirely priors (patent counts,
literature, library hits), not better spectra. Transfer: any allowed prior (train-frequency,
candidate source counts, spectral-library cosine) belongs in the reranker with a large weight. The
competition-deep-dive doc should confirm whether test molecules are "dark" (priors ≈ 0, skip) or
known-ish (priors = free LB).

### 5. Spectral-library exact match first, prediction second (SIRIUS ZODIAC pattern)
SIRIUS 6 pipeline: high-cosine library hits become fixed "anchors" that override prediction;
CSI:FingerID only ranks the remainder. CASMI 2022's best structure entries all fused library +
prediction. Transfer: stage 0 = cosine search of each query spectrum against all train spectra;
copy the winner's structure at high cosine into rank 1-3 before neural scoring. Costs minutes,
protects the easy points our v0 currently drops.

### 6. Adduct handling is a silent killer (CASMI 2022: 13-94% spread across teams)
Worst teams got 13-18% of adducts right; SIRIUS 93-96%. Wrong adduct = wrong mass = truth never
enters the candidate set = MRR 0 regardless of scorer. Transfer: explicit adduct classifier
([M+H]+ vs [M+Na]+ vs [M+K]+ etc.) before candidate pull, with K+/Na+ rule backstops (data-driven
models underperform rare adducts — MIST-CF Fig 3e, already noted). Cheap insurance.

### 7. Multi-energy / multi-spectrum fusion is our unique edge (nobody in CASMI had 1-9 spectra)
CASMI scored one spectrum per challenge; we get up to 9 per molecule. DrivenData Mars winners show
the pattern: ensembles over diverse preprocessing configs of the same signal beat any single model.
Transfer: attention/mean-pool spectra per molecule weighted by base-peak intensity (already in v1
plan); additionally vary peak-picking thresholds per branch and fuse. No literature to copy — must
validate locally.

### 8. Dedup + candidate hygiene is load-bearing at 25 slots
CASMI papers note tautomer/stereo duplicates scatter votes across identical answers; MassSpecGym caps
at 256 same-mass candidates. Transfer: InChIKey-14 dedup before ranking, mass-filtered candidate
sets only, fill all 25 slots (empty slots = certain zero). Already in v1 plan; listed here because
CASMI history shows teams losing medals to it.

### 9. Class prediction (CANOPUS-style) as a rerank feature, not a target
SIRIUS stack predicts compound class from spectrum and uses it to gate candidates; CASMI 2022 scored
class separately (winners ~70-75%). Transfer: add predicted-class vs candidate-class agreement as one
LightGBM feature. Do not chase class accuracy itself — metric pays only exact structure.

## What is obsolete (do not copy)
- Manual/expert interpretation (won 2012-2013): no human loop here.
- Single-tool MetFrag/MAGMa-only pipelines: beaten by fingerprint methods since 2015-2016.
- Exact-mass PubChem enumeration per query: infeasible at our scale; mass-filtered isomer sets + top-256 funnel instead.
- De novo generation as primary output (MSNovelist/MS2Mol-style): CASMI 2022 exact-structure rate
  for unknowns was ~25-30% WITH retrieval; pure generation Top-1 ≈ 0 (MassSpecGym confirms). Use
  generative only as slot-fillers 16-25.
- GC-MS/Mars-spectrometry 2D-CNN image tricks: different instrument physics (temperature-ramp GC),
  not LC-MS/MS; only the ensemble-diversity lesson transfers.

## Suggested build order from this survey (maps to v1 plan)
1. Formula ranker + adduct classifier + library-anchor stage 0 (findings 1, 5, 6).
2. Fingerprint scorer as primary ranker (finding 2).
3. One diverse second scorer + LightGBM/vote fusion (finding 3) + metadata priors if rules allow (finding 4).
4. Per-molecule multi-spectra fusion + dedup/fill-25 hygiene (findings 7, 8); class-agreement feature (finding 9).
