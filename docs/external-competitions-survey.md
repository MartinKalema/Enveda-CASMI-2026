# External competitions survey: everything outside CASMI/MassSpecGym/MIST/Mars

Branch: external-competitions survey | Date: 2026-09-22 | Scoreboard: v0 0.088 vs top 0.409
Companion docs: `docs/constraints-and-literature.md`, `docs/casmi-history-survey.md`
(CASMI contests, MassSpecGym, MIST/MIST-CF, MS2Mol, DrivenData Mars 1+2 already covered — not repeated here.)

Scope searched: DrivenData, Zindi, DREAM/Synapse, Codalab/Codabench, AICrowd, Tianchi/Alibaba,
EvalAI, Grand Challenge, NeurIPS competitions, Kaggle chemistry, ASMS / Metabolomics Society /
NORMAN / EPA exposome-NTA trials, drug-metabolite prediction. Read-only research; nothing committed
except this file (uncommitted until user approves).

## Ranked findings (most transferable to our MRR@25 first)

### 1. EPA ENTACT — the closest real-world analog to our test set (blinded unknowns in mixtures)
Organizer/year: US EPA, Non-Targeted Analysis Collaborative Trial, 2018-2020 (Sobus/Ulrich et al.,
Anal Bioanal Chem; follow-ups Chao et al. 2020, McEachran et al. 2019).
Task + metric: identify spiked unknowns in synthetic mixtures ("pass/fail" per compound, blinded then
unblinded scoring). ~1,269 ToxCast compounds across 10 mixtures; HRMS (LC + GC).
What won / key numbers: no single ML winner — it is an interlab trial, but the head-to-head numbers
are gold. Reference-library match + CFM-ID in-silico spectra correctly identified ~53% of the 377
compounds with acquired MS2 (Chao et al.). Standalone fragmenters (MetFrag, MAGMa) far behind;
CSI:FingerID-style fingerprint methods ~39% top-1 vs ~half that for pure fragmenters on the same
mixtures. Blinded true-positive rate roughly DOUBLED after unblinding (i.e., priors/metadata dominate).
Dataset/domain: environmental/exposome chemicals (ToxCast), LC-HRMS + GC-HRMS — broader and more
"dark" than our natural-product test, but same blinded-unknown regime.
Transfer verdict: HIGH. Confirms the CASMI-history findings with independent numbers: (a) library
cosine first, fingerprint prediction second, fragmentation simulation only as backup; (b) expect our
ceiling on truly novel natural products to be ~30-55% top-1, so MRR comes from getting the easy
library-anchor points cheap and spending model capacity on ranking, not recall; (c) any allowed
metadata/library prior gets a large rerank weight (ENTACT unblinded doubling).

### 2. DREAM Olfaction Prediction (2015, Keller/Vosshall, Synapse) — small-data lesson
Organizer/year: DREAM / Rockefeller (Keller, Vosshall), 2015; Science 2017 paper (cited 460+).
Task + metric: predict human smell ratings (19 descriptors, 49 subjects) from molecular structure;
~500 molecules. Pearson correlation per descriptor.
What won: ensemble of random forests on Morgan-fingerprint + Dragon physicochemical descriptors.
Deep nets did NOT win — with ~500 training molecules, descriptor engineering + bagged trees beat
every neural entry. Key trick: per-descriptor model selection (different descriptors wanted different
features), then averaged.
Dataset/domain: structure->perception, not spectra — but the data regime matches our bottleneck
(small labeled set, large chemistry space).
Transfer verdict: HIGH for the reranker stage. Our LightGBM rerank (neural score, mass ppm,
Tanimoto, neutral-loss count, library prior) is exactly the "descriptors + bagged trees on small
data" pattern that won here. Do NOT try to learn the final ranking end-to-end from spectra alone;
keep a GBM-over-features final stage. Also copy per-target heads (CHAMPS finding 4 reinforces this).

### 3. Kaggle Merck Molecular Activity (2012) — why multitask fingerprint DNNs work
Organizer/year: Kaggle + Merck, 2012 ($40k). Task: predict 15 assay activities from structure
(QSAR), ~160k compounds; metric R^2-ish per-assay average.
What won: George Dahl (Hinton group) — multitask deep net over ECFP fingerprints, first famous
"deep learning wins Kaggle chemistry" result. Key trick: shared hidden layers across 15 assays =
auxiliary-task regularization; single-task nets overfit.
Dataset/domain: drug-like synthetic molecules, structure-in (no spectra).
Transfer verdict: MEDIUM-HIGH. Direct justification for multitask fingerprint prediction as our
primary scorer: predict a 1024-bit fingerprint (or per-bit groups) from subformula-encoded spectra
with shared trunk, exactly as MIST does. If we add auxiliary heads (formula, adduct, compound
class/CANOPUS-style), expect the same regularization lift Dahl got. No spectra involved, so the
spectral encoder itself must come from MIST-style pretraining, not from this.

### 4. OGB-LSC PCQM4Mv2 (NeurIPS 2021/2022) — the graph-encoder recipe if we need molecules-as-graphs
Organizer/year: Stanford OGB Large-Scale Challenge @ NeurIPS/KDD 2021-2022. Task: HOMO-LUMO gap
from 2D graph, 3.8M molecules; metric MAE.
What won: ensembles of hybrid MPNN/Transformer (GPS++, TokenGT, Transformer-M, VisNet). Key
tricks, all documented: (a) generate 3D conformers (RDKit) and use as auxiliary supervision even
though test input is 2D-only; (b) large-scale pretraining then finetune; (c) ensemble of 5-7 seeds/
hyperparams, not architectures.
Dataset/domain: 3.8M drug-like/QM molecules, structure-in.
Transfer verdict: MEDIUM. Relevant only where we encode candidate structures (rerank side): use
pretrained graph/fingerprint encoders, generate RDKit conformers for auxiliary features, and
ensemble seeds rather than stacking diverse architectures (pairs with CASMI finding 3: two diverse
scorers max, then seed-ensemble each). Do not burn 9h budget training graph nets from scratch.

### 5. Kaggle CHAMPS Scalar Coupling (2019) — per-type heads + physics hybrid
Organizer/year: Kaggle + CHAMPS (Quantum Uncertainty), 2019 ($30k). Task: predict NMR scalar
coupling constants; metric log-MAE per coupling type (8 types).
What won (#1, "Quantum Uncertainty" team): per-type output heads (no shared interaction layers
across types — sharing HURT), whole-molecule transformer encoder, ensemble with a physics-based
model (hybrid ML + DFT-derived features). 9th/10th places confirm: separate heads per type and
multi-task energy-style auxiliaries.
Dataset/domain: QM-computed NMR on small organics — spectroscopy-adjacent (like ours, signal
determined by structure + physics), but NMR not MS/MS.
Transfer verdict: MEDIUM. Two concrete transfers: (a) per-adduct / per-instrument heads instead
of one shared scorer head (rare adducts underperform shared models — same failure MIST-CF Fig 3e
shows for K+); (b) hybrid neural + physics features in the reranker (neutral-loss masses, RDBE,
isotope-pattern score alongside neural scores). Validates the v1 LightGBM-over-mixed-features plan.

### 6. Kaggle BMS Molecular Translation (2021) — TTA + seed-ensemble discipline
Organizer/year: Kaggle + Bristol-Myers Squibb, 2021. Task: image-of-molecule -> SMILES
(OCR-for-chemistry); metric Levenshtein distance.
What won: vision-transformer encoder-decoders with heavy test-time augmentation (rotations/
shears of the input image) + seed ensembles. Key trick was not architecture but TTA: 5-10
augmented views per test input, averaged — worth more than any backbone swap.
Dataset/domain: rendered structure images; no spectra.
Transfer verdict: MEDIUM-LOW, one trick transfers directly: test-time augmentation over spectra.
Vary peak-picking thresholds / noise masks per branch (already proposed in casmi-history finding 7)
and average — the BMS winners show TTA beats backbone upgrades, and it costs only inference time
inside our 9h kernel. Also: decoder-validity handling (valid-SMILES约束) matters for our fill-to-25
slots if we use generative fillers.

### 7. NORMAN Network suspect/non-target screening trials (2015-ongoing) — priors win in exposome ID
Organizer/year: NORMAN Association (EU), collaborative suspect-screening trials on water/sludge
HRMS samples. Task: identify environmental contaminants from LC-HRMS; metric detection/ID rate.
What won (recurring result across trials): database/suspect-list-constrained workflows
(MetFrag + CompTox dashboards + retention prediction) beat open-ended elucidation by 2-3x;
retention-time prediction as an orthogonal filter was the single most-cited "free" gain.
Dataset/domain: pollutants/exposome, LC-HRMS — mass-filtered candidate logic identical to ours.
Transfer verdict: MEDIUM. Reinforces: constrain-then-rank (suspect list = our mass-filtered
candidate set), and add an orthogonal cheap filter. We have no retention time, but the analog is
collision-energy / instrument-covariate consistency + isotope-pattern match as veto features in
the reranker. If test metadata includes retention or CCS, use it as a hard filter.

### 8. FlexMS benchmark + MS-BART (NeurIPS 2024/2025) — newest multi-task pretraining signal
Organizer/year: academic (arXiv 2026 FlexMS; NeurIPS 2025 MS-BART poster). Task: unified
spectrum<->molecule modeling; FlexMS reuses CASMI challenges as the "real-world retrieval" split.
What won: MS-BART multi-task pretraining (fingerprint + structure objectives jointly) over
single-task encoders; cross-modal objectives beat per-spectrum contrastive alone.
Dataset/domain: public MS/MS libraries, CASMI-derived eval.
Transfer verdict: MEDIUM-LOW (bleeding edge, code immature). Only transfer: if v1 plateaus,
pretrain the spectral encoder multitask (fingerprint + formula + adduct heads), not contrastive-only.
Do not build on FlexMS/MS-BART code directly unless v1 + rerank saturate.

## Negative results (searched, nothing transferable — do not spend more time here)
- Zindi, Tianchi/Alibaba, AICrowd, EvalAI, Grand Challenge: no MS/MS small-molecule-ID
  competition found on any of these platforms (Grand Challenge is medical-imaging-only; Zindi/Tianchi
  chemistry = tabular QSAR or NLP-for-chemistry; AICrowd/EvalAI chemistry = molecule generation /
  retrosynthesis, no spectra). Closest items are QSAR-style (see Merck/DREAM above).
- ASMS / Metabolomics Society: no formal leaderboard challenge exists; society "challenges" are
  workshop tutorials or position papers (Metabolomics Society ID Task Group publishes benchmark
  datasets, not contests). Nothing to copy beyond ENTACT/NORMAN numbers above.
- Drug-metabolite prediction as a contest: no open leaderboard found (Meteor/Lhasa commercial,
  CYP-site prediction is literature-only). The per-task lesson is captured by DREAM Olfaction
  (small-data descriptors win) — no separate entry needed.
- Codalab/Codabench: hosts CASMI-adjacent academic evals but no standalone small-molecule-ID
  contest with published winner write-ups beyond what CASMI papers already report.

## Build-order deltas from this survey (amendments to v1 plan, not replacements)
1. Keep formula-first + fingerprint-primary + 2-scorer-consensus + GBM rerank (confirmed by
   ENTACT independent numbers, not just CASMI).
2. Reranker MUST be descriptors + bagged trees/GBM on mixed neural/physics/prior features
   (DREAM Olfaction + CHAMPS); add per-adduct heads and isotope-pattern + instrument-consistency
   veto features (CHAMPS + NORMAN).
3. Add spectral TTA (threshold/noise variants, averaged) — cheapest inference-only gain (BMS lesson).
4. Multitask spectral encoder (fingerprint + formula + adduct + class heads, shared trunk) as the
   v1.1 upgrade if single-task fingerprint plateaus (Merck + MS-BART lessons).
5. Expectation-setting: blinded-unknown top-1 ceiling ~30-55% (ENTACT/CASMI); MRR@25 comes from
   library anchors + rank-1 precision, not from chasing recall.
