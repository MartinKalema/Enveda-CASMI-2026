# Spectrum-to-structure SOTA 2023–2026: papers, verdicts, new designs

Prior docs cover MIST / MIST-CF / MS2Mol / MassSpecGym / FraGNNet basics
(`docs/components/fingerprint-model.md`, `docs/components/pool-frag-validation.md`).
This doc goes beyond: 9 PDFs in `docs/papers/structure/` (all curl-downloaded,
`file`-verified; `.txt` = pypdf extraction for grep, `-clean.txt` = control-char
stripped for the reader), plus cited-only works that are paywalled or blocked.

Downloaded (all PDF, verified): JESTR (arXiv 2411.14464), DiffMS (arXiv
2502.09571), FraGNNet (arXiv 2404.02360), ICEBERG (arXiv 2304.13136), MS-BART
(arXiv 2510.20615), GLMR (arXiv 2511.06259), MARASON (arXiv 2502.17874), MS-GPT
(arXiv 2607.23607), FLARE (bioRxiv 10.64898/2026.01.27.702086, 26pp).
NOT downloadable: MVP (ACS Anal Chem 2026, paywall), CMSSP (ACS 2024, paywall),
CFM-ID 4.0 papers (paywalled; site + baseline roles used instead), MS-CLIP
(OpenReview, HTML only).

## 1. Per-paper verdicts (retrieval-first lens: exact InChIKey match only)

### 1A. Contrastive spectrum–molecule models

**FLARE (Chen et al., bioRxiv Jan 2026) — VERDICT: highest-value architecture.**
Replaces global cosine with bidirectional peak↔atom MaxSim (FILIP-style):
spectrum = set of subformula-annotated peaks → linear + transformer; molecule =
GNN atom reps; score = mean(row-max, col-max cosine). Physical (not semantic)
weak supervision; alignments inspectable vs MAGMa. MassSpecGym Table 1
(same protocol): mass-based rank@1 **43.15** / @5 75.59 / @20 92.89
(vs MVP 26.37, JESTR 15.13, MIST 14.64); formula-based **22.66** / 50.00 / 75.15
(vs MVP 11.10, JESTR 11.85). Killer detail §2.4: FLARE vs JESTR rank Spearman
**0.34**, top-10 Jaccard **0.27** — the two paradigms are complementary, and
FLARE wins by larger margins on molecules *dissimilar* to train (Q1 similarity
bin). Authors themselves motivate a hybrid. Transfer: the MaxSim head is a
cheap add-on to any dual encoder; subformula peak annotation reuses
`v1/subformula.py`. Caution: absolute numbers shift across MassSpecGym
versions (MVP 26.37 here vs 14.0 elsewhere) — re-eval locally, trust deltas.

**JESTR (Kalia et al., Bioinformatics 2025) — VERDICT: the recipe baseline.**
GNN molecule encoder + 1-Da-binned 1000-dim 3-layer MLP spectrum encoder,
CMC/InfoNCE τ=0.05, batch 32, 1000 epochs (A100; NPLIB1 train 6h, infer 1.5h).
Key invention: **candidate regularization** — last 3% epochs, α=0.9/β=0.1,
minimize cosine of spectrum vs top-32 PubChem same-formula candidates by
Tanimoto. MassSpecGym: mass 15.13/36.75/60.32, formula 11.85/32.95/61.46;
NPLIB1 rank@1 45.76 (+reg) vs 41.06 (−reg); NIST2020 38.62 vs 36.38.
Regularization helps most at rank@1, i.e. exactly our metric. Transfer: copy
the loss + late-turn-on hard-decoy schedule verbatim; spectrum encoder is
deliberately weak (bins) so our set-transformer input is a strict upgrade.

**MVP (Chen et al., Anal Chem 2026; via FLARE Table 1 + fingerprint-model.md) —
VERDICT: cited-only, multi-view contrastive.** Multi-view spectra × multi-view
molecules; FLARE Table 1 mass 26.37/58.90/86.88 (2nd best), formula
11.10/31.14/61.99. Lesson: extra views help mass-based far more than
formula-based — mirrors our dual single/merged-view finding. No PDF (paywall);
replicate the *idea* (multi-view InfoNCE), not the paper.

**CMSSP / MS-CLIP / CSU-MS2 (cited-only) — VERDICT: existence proofs.**
CMSSP (dot-product discriminator, GNPS+MassBank trained) collapses under
retraining (JESTR Table 2: retrained 13.92 vs released 54.09 on NPLIB1 —
pretraining-data leakage, not architecture). MS-CLIP = modality-shared CLIP
for spectra. CSU-MS2 = sinusoidal m/z + attention spectral encoder. Lesson:
leakage-audit every pretrained weight (our fork-034 lesson 11 rhymes).

### 1B. Diffusion / generative elucidation (de novo; exact-match lens)

**DiffMS (Bohde et al., ICML 2025) — VERDICT: best honest de novo baseline.**
MIST formula-transformer encoder (SIRIUS peak formulae + pairwise neutral
losses) → formula-restricted discrete graph diffusion decoder (DiGress-style
marginal-noise schedule, cosine; Graph Transformer predicts A₀; heavy atoms
fixed from formula, H implicit). Decoder pretrained on **2.8M fp–molecule
pairs** (DSSTox/HMDB/COCONUT/MOSES, test-scaffold-excluded), encoder
pretrained spec→fp, end-to-end finetune. NPLIB1 top-1 **8.34%** (MIST-CF
formula; 7.03% true? no — 8.34 true / 7.03 MIST-CF); MassSpecGym **2.30%**
true formula / 1.86% MIST-CF. Ablations: encoder pretraining ~2× top-1;
decoder pretraining scales monotonically with data (0→2.8M); formula error
costs only ~0.4pp. Transfer: formula conditioning is load-bearing (our failure
mode 7); the "pretrain decoder on infinite cheap fp–molecule pairs" trick is
directly reusable for any filler model. For exact-match retrieval: ~0 LB —
generators don't rank our 56-candidate windows.

**MS-BART (Han et al., NeurIPS 2025) — VERDICT: cheaper DiffMS rival, same
ceiling.** BART-base from scratch; spectra → MIST-predicted formula-conditioned
4096-bit Morgan fp (threshold ε=0.2 NPLIB1 / 0.11 MSGym) → `<fpNNNN>` tokens +
185 SELFIES tokens; 3-stage: (1) 4M fp↔SELFIES pretrain (denoise + translate +
hybrid, MCES≥2 from test), (2) finetune on experimental, (3) freeze encoder +
Tanimoto-rank contrastive alignment (anti-hallucination). NPLIB1 top-1 7.45%
(< DiffMS 8.34 — authors admit DiffMS's leakier pretrain exclusion); MSGym
1.07% (Na-adduct train scarcity: 15.5% of train; they filter Na at finetune,
keep at test). Gold-fingerprint ceiling: 73.5% NPLIB1 / 47.6% MSGym top-1 —
the fp→structure map is nearly solved; the spectrum→fp step is the whole
game. Transfer: validates our fingerprint-channel-first strategy; rank-loss
alignment is a portable finetune trick.

**MS-GPT (Zhao et al., arXiv Jul 2026) — VERDICT: most transferable idea,
most discounted numbers.** Reframing: don't threshold the posterior once —
*query a band of it*. SAFE-GPT backbone + fp/formula cross-attention,
molecule-only pretrain (~100M structures, 2 epochs, formula-grouped batches),
active-bit density-band calibration (match thresholded query density to
oracle density, sweep ρ≈0.95–1.31), M queries × N samples, InChIKey-14
frequency-consensus ranking, LoRA only on query-reading pathway (+ frozen
KL anchor, recoverability-weighted). Claims NPLIB1 29.8/41.1, MSGym
23.9/28.7 top-1/10 — but **known-formula protocol** throughout (Table 1
caption). Still ~10× DiffMS true-formula 2.3%, unexplained by formula alone;
treat as unverified until reproduced (possible split/protocol difference).
Steal regardless: (a) density-band multi-query + frequency consensus is a
drop-in upgrade for ANY fp→candidate step (ours included); (b) formula-grouped
batching (same-formula isomers in contrast) ports to contrastive training;
(c) LoRA-on-readout adaptation is the cheapest domain-shift fix.

**MSNovelist follow-ups (MIST+MSNovelist, Neuraldecipher, MolForge, MADGEN,
FRIGID/MBGen — via DiffMS/MS-BART/MS-GPT baselines) — VERDICT: slot-fillers.**
Retrained MIST+MSNovelist: NPLIB1 5.40% top-1, MSGym **0.00%**. Corrected
MIST+MolForge: 10.7%/14.5% top-1/10 (the 28%/36% was the batch-mask bug —
see fingerprint-model.md §C). FRIGID: strongest exact-match baseline per
MS-GPT (their Table 1). MADGEN (scaffold-retrieval + RetroBridge): 1.3% MSGym.
Lesson: all de novo exact-match ≈ 0–2% without known formula + giant
pretraining → never a ranker, only tail-slot fillers (our slots 16–25).

### 1C. Retrieval-augmented generation

**GLMR (Zhang et al., AAAI 2026) — VERDICT: do not plan on it (audit flag
stands).** Pre-retrieval (ChemFormer SMILES encoder frozen + tuple-(m/z,int)
transformer spectrum encoder, dual InfoNCE) → top-K priors → cross-attention
fusion → ChemFormer decoder generates molecule → re-rank candidates by
mol–mol cosine. Claims MSGym Recall@1 **64.2/68.5** (weight/formula) vs JESTR
17.6/11.8 — a 3–6× jump *over its own JESTR reproduction*, irreconcilable
with JESTR/FLARE/MVP literature (15–43%). New MassRET-20k (12 adducts, full
NCE) is a genuine contribution; the 64% is not — likely protocol/leakage
difference (same class of artifact as the 2026 audit's PubChem-ranking bias).
Steal the shape (retrieve→generate→unimodal re-rank), never the numbers;
require spectrum-blind baseline before trusting any such claim.

**MARASON (Wang et al., ICML 2025) — VERDICT: the principled RAG lesson.**
Forward direction (structure→spectrum): retrieve train-set reference by Morgan
Tanimoto (same adduct/instrument, ≤3 CEs) → fragment both with ICEBERG →
**neural graph matching** (3 GNNs + Sinkhorn/softmax assignment) aligns
fragment DAGs → transformer predicts intensities. NIST20 top-1 retrieval
19%→**28%**; naive concat-RAG ≈ no gain. Lesson for us: *how* you fuse the
retrieved neighbor matters more than retrieving it — explicit alignment
(graph matching / cross-attention over fragments) beats concatenation. The
retrieval DB (train spectra) mirrors our analog channel; the matching module
is the upgrade path for analog fusion (cf. Design C).

### 1D. Forward / in-silico fragmentation SOTA

**ICEBERG (Goldman et al., 2024; +2.1 2026 w/ GPU speedup, NIST23 weights) —
VERDICT: offline-distill, don't infer.** Generate (GGNN predicts per-atom
breakage, MAGMa-DAG supervision, atom- not bond-level so every event changes
heavy composition) → Score (Set Transformer, ±H-shift intensities). NPLIB1
cosine 0.727 vs SCARF 0.726 / MassFormer 0.721 / NEIMS-GNN 0.694 / CFM-ID
0.412; NIST20 0.699. Top-1 retrieval lift on NIST20 "not pronounced" —
forward accuracy ≠ ranking accuracy (same Pareto lesson as the BCE finding).
Practical: `coleygroup/ms-pred` + pretrained NIST23 model; precompute
fragment bags per pool structure offline, ship `.npy`, lookup in-kernel
(pool-frag-validation.md upgrade 7 stands; cost it as one GPU-day).

**FraGNNet (Young et al., TMLR 2025; basics in pool-frag doc) — VERDICT:
best NIST20 forward numbers + annotation discipline.** Recursive heavy-atom
DAG (d=3/4, H-tolerance j=4 covers most intensity) + GINE mol/fragment GNNs
(gFRAG→node-MLP ablation: edges droppable, faster) + P(n)P(f|n) probabilistic
head, Gaussian-mass mixture, OS-probability loss term, latent-entropy
regularization. NIST20: CBIN 0.736/0.678 (InChIKey/scaffold) vs ICEBERG
0.707/0.636; retrieval top-1 36.0/30.3 (ties/beats ICEBERG, bigger gap on
scaffold); annotation F1 0.89 vs ICEBERG 0.79 / GrAFF 0.74; fragment-mode
agreement PFA≥82–91% across entropy settings. For us: (a) preferred
distillation source alongside ICEBERG (higher scaffold-split fidelity =
better NP transfer); (b) entropy-regularization analysis is the template for
auditing our own alignment heads.

**CFM-ID 4.0 (Wang et al., Anal Chem 2021 + NAR 2022 server) — VERDICT:
rule/EM baseline, keep as sanity check only.** Competitive fragmentation
modeling (EM over breakage transitions); 4.0 adds improved fragmentation
probabilities + double-bond breaks + larger experimental/in-silico DB;
source + Docker available. Quantitatively lapped (0.41 cosine vs 0.70+).
Its surviving value: hand-tuned bond-energy priors that MetFrag-lite dropped
— reimport BDE weights + depth-3 as the cheap in-kernel upgrade (pool-frag
upgrade 3), not the model itself.

## 2. Cross-cutting lessons (new vs prior docs)

1. **Global ⊕ fine-grained ensemble is the free lunch.** FLARE↔JESTR
   Spearman 0.34 / Jaccard 0.27; each wins big subsets. Two heads, one
   encoder, score-fusion — expect +0.01–0.02 retrieval on top of either.
2. **Never threshold a posterior once** (MS-GPT): sweep the density band,
   pool, frequency-consensus. Applies to our fp bits and to MIST-CF formula
   lists alike.
3. **Formula-known vs formula-predicted is a 10× numbers gap** (MS-GPT 23.9
   vs DiffMS 2.3). Discount every "SOTA generation" claim by its formula
   protocol; our setting is formula-predicted.
4. **Absolute retrieval %s are incomparable across papers** (MVP 11–26,
   FLARE 22–43, GLMR 64–68 on "the same" benchmark). Candidate construction
   and split version dominate. Local disjoint@5 is the only currency.
5. **Generative exact-match without known formula + ≥3M pretraining is
   0–2%.** Generators are recall fillers (ρ₂ lever), never rankers.
6. **RAG needs alignment, not concatenation** (MARASON +28% vs naive ~0).
7. **Forward-model cosine ≠ retrieval accuracy** (ICEBERG) — distill
   fragments as features, don't expect the simulator to rank.

## 3. New designs, ranked (all beat fingerprint-MLP-regression on the metric)

### Design 1 (best bet): Dual-head contrastive retrieve
...[truncated 3881 chars]