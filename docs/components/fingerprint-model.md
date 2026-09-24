# Channel-4 fingerprint model: fork-034 FPNet, why it wins, SOTA upgrades, our gap, plan

## 1. Their exact model (all refs = `kernels/fork-034/notebook.ipynb` cells)

**Role in pipeline.** Per-molecule 4th evidence channel feeding the 31-feature GBM ranker (cell 8 `rank_features`, cell 9 inference).
Alone 0.468 MRR, +analog 0.567, all-four 0.612; V17 Class-1 0.873 / Class-2 0.612, public LB 0.300–0.336+ (cell 0 milestone table).
Weights: `prvsiyan/casmi26-fp-models-v2` (`fp_*.pt`; `single` vs `merged` by filename) — not downloadable offline here, so architecture/training below is from CODE (cell 7) + survey measurements (`docs/kaggle-notebooks-survey.md` findings 5, 10–11), not weight inspection.

**Architecture — `FPNet` (cell 7).** `FPNet(nbits, d=512, layers=6, heads=8, drop=0.1)` (ckpt stores `nbits/d/layers/step`; `load_neural_models` rebuilds from ckpt).
Per peak: sinusoidal m/z embedding + sinusoidal neutral-loss embedding (`nl = prec − mz`, clamped ≥0) + raw intensity → `Linear(2d+1 → d)` (`self.pk`).
Global vector: sinusoidal precursor-m/z + `[CE/100, polarity(±1), log1p(prec)/10]` linear + learned adduct embedding (26 classes in `ADDUCT_LIST`, incl. dimers `[2M+H]+/[2M+Na]+`, `<unk>`) + instrument embedding (5 families via `instr_family`: timsTOF / Orbitrap-QFT / QTOF / trap-QQ / other).
Sequence = `[global, peak₁…peak_N]` through 6 pre-norm transformer `Block`s (QKV attention + GELU FFN 4d), LayerNorm, then `head([CLS; masked-mean]) → 2048 → GELU → Dropout → nbits` raw logits `z`.
Target: **6,930 informative bits** filtered from `ECFP4 + ECFP6 + RDKitFP + MACCS` (survey §"one family"; cell 5 `fp_and_mass` builds the same concat then `[BITS]` mask). NOT plain Morgan-2048.

**Inputs / preprocessing — `prep_peaks` (cell 7).** Drop peaks with `mz > prec+1.5`; floor 0.1% base peak; cap 128 peaks with per-50-Da-window cap 8 (keeps weak-but-isolated peaks); sort by m/z; intensity = `sqrt(I/Imax)`.
Covariates per spectrum: adduct, instrument family, CE (mean, fallback 25 eV), polarity.

**Dual views — `_logits_from` + `_merge_peaks` + `model_logits` (cell 7).**
Per-spectrum view: each of the molecule's 1–9 spectra → FPNet → mean over spectra (single-model files).
Merged view: all spectra peak-lists concatenated, re-normalized per spectrum, 0.005 Da dedup keeping max intensity → one FPNet pass at median precursor m/z, CE=25, mean polarity (merged-model files).
Final `z = mean(single-view, merged-view)`. Rationale: per-spectrum view preserves CE/adduct diversity; merged view boosts S/N on shared fragments. Our survey notes per-molecule fusion is our structural edge (papers are per-spectrum).

**Ranking — `f·z` raw-logit dot (cells 7–8).** `score = cf @ z` with `cf` = 0/1 candidate bits, `z` = raw logits — the exact Bayes log-likelihood over independent bits (no sigmoid/calibration; survey finding 5).
Cell 8 turns it into 6 GBM features: raw `f·z`, z-score, rank-norm, `raw − max`, normalized variant, is-max flag — plus 6 cross-agreement features (`lib·(1−rank)`, `analog·(1−rank)`, agreement scalars, `corrcoef`).

**Training recipe (from survey finding 5 + `v-fpnet` replication notes; training script lives in the weights author's private kernel, not in cell 7).**
Softmax CE over truth + **63 same-±10 ppm-window decoys**; **peak dropout + intensity jitter + ±5 ppm m/z noise** augmentation; **best-by-validation-retrieval checkpointing** (they memorize: top-1 0.457@12k → 0.127@69k steps while loss keeps improving — loss lies, keep the early checkpoint).
Candidate pool: 711,705 structures (COCONUT 2.0 462k ∪ train 275k, InChIKey14-deduped, mass-sorted; cell 5 `CandidatePool`), window ±8.5 ppm (median 56 candidates, cell 1 `CFG`).

**Why THAT model and not alternatives.** MIST-lineage subformula/set transformers beat binned FFNs 14.6% vs 2.5% hit@1 on MassSpecGym (MassSpecGym-SPOTLIGHT numbers in `docs/constraints-and-literature.md`); the gap comes from domain encoding (subformula/neutral-loss peaks), not depth. `f·z` is the Bayes-exact scorer for bit logits, so no calibration layer can beat it on matched bits. Dual views are the only component that exploits our 1–9 spectra/molecule structure. Alternatives were measured and rejected: PubChem expansion 0.52→0.35 (finding 3), consensus fingerprint 0.43 vs 0.52 max (finding 13), DreaMS-embeddings analog channel no-win (finding 11), de novo generators ~0 exact-match (finding 14).

## 2. Assumptions + failure modes

1. **Truth-in-window.** ±8.5 ppm with 30 ppm fallback; precursor/adduct error (Na↔H swaps, riken 0.005 Da rounding — finding 8) empties the window → channel scores nothing. Mitigate with `max(10ppm, 0.01Da)` + formula-mass index.
2. **Adduct/instrument coverage.** 26 adducts + 5 instrument families; rare adducts (K⁺) and unseen timsTOF variants fall to `<unk>`/other embeddings (MIST-CF Fig 3e pattern).
3. **Bit-independence.** `f·z` ignores bit correlation; ECFP6/RDKit bits are highly correlated → overconfident scores on large near-duplicate scaffolds. The GBM rank-norm features partially absorb this.
4. **Memorization.** Fingerprint nets memorize libraries (finding 5); public weights saw most libraries (finding 11: 0.49 held-out vs 0.76–0.82 seen). Trust the channel most inside the GBM blend, never standalone.
5. **Merged-view fragility.** One bad spectrum (wrong precursor, chimeric) pollutes the merged peak-list; per-spectrum mean is the backstop, not vice versa.
6. **Decoy-distribution shift.** Trained vs ±10 ppm COCONUT+train decoys; test uses same pool so shift is small — but any pool change (Bio-DB add) dilutes without retraining (finding 3/13).
7. **No formula conditioning.** FPNet predicts all 6,930 bits unconditioned; wrong-formula candidates can still outscore on shared scaffolds. Formula-gated rerank (MIST-CF) is the missing complement.

## 3. SOTA 2024–2026: what beats their FPNet, and what to steal

Baseline reminder (MassSpecGym retrieval, no formula): FFN 2.5 / DeepSets+Fourier 5.2 / **MIST 14.6% hit@1**. Everything below is post-MIST.

**A. Joint-embedding retrieval beats fingerprint regression — the paradigm shift.**
- **JESTR (Bioinformatics 2025):** spectrum encoder + molecule encoder (graph+fingerprint) into one space via CMC/InfoNCE (temperature-scaled cosine) + **candidate-molecule regularization** (train-time same-formula decoys, +5.7% rank@1). Beats spec-to-FP/mol-to-spec by 55–300% avg rank@1–20. Recipe to copy: dual encoders, InfoNCE over in-batch + hard same-mass negatives.
- **MVP (2025):** multi-view spectra × multi-view molecules contrastive; 2nd-best corrected model (H@1 14.0 / H@5 36.9 / H@20 68.1 with formula bonus).
- **FLARE (bioRxiv Jan 2026):** replaces global cosine with **bidirectional peak↔atom fine-grained alignment** (GNN atom reps, physical weak supervision) + contrastive loss. SOTA on corrected MassSpecGym: H@1 **22.7** / H@5 50.0 / H@20 75.2 (bonus), +63% rank@1 over MVP, +195% over MIST. Inspectable alignments = debuggable failures. **Highest-value architecture to replicate.**
- **FRIGID / MIST-vFRIGID (2025–26):** current corrected-library leaders (MIST-vFRIGID H@1 53.8 / H@5 65.3; FRIGID generative 45.3/58.3). Details thin; treat as existence proof that MIST-encoder + modern retrieval training still has headroom.
- **Loss-tradeoff study (arXiv 2602.16507):** systematic MassSpecGym loss comparison — **contrastive Emb-Cos best retrieval** (H@1 ~12–13.6), **IoU/Focal best fingerprint Tanimoto**; BCE optimizes neither; retrieval accuracy and fingerprint fidelity form a **Pareto front**. Implication: stop tuning BCE Tanimoto; train the ranker loss (contrastive `f·z`/cosine) directly.

**B. Formula-conditioned models (fix failure mode 7).**
- **MIST-CF (JCIM 2024):** energy-based formula+adduct ranker over ≤256 candidates, same Formula-Transformer backbone; 0.769 top-1, tied SIRIUS CASMI2022 0.868 joint. Recipe: subformula peaks (top-20, ≤15 ppm, RDBE filter) + instrument/adduct covariates + softmax over true+hard decoy formulae. Build as a pre-filter + GBM feature.
- **MS-BART (NeurIPS 2025):** fingerprints as modality-invariant tokens → BART-base pretraining on 3.6M fingerprint↔SELFIES pairs (denoise+translate), finetune on **MIST-predicted (formula-conditioned) fingerprints**, + frozen-encoder chemical-feedback alignment. SOTA 5/12 metrics, 10× faster than diffusion. Recipe: massive cheap fp→structure pretraining, then align to noisy predicted fps.
- **DiffMS (ICML 2025):** MIST formula-transformer encoder (peak formulae + pairwise neutral losses) → **formula-restricted discrete graph diffusion decoder**, decoder pretrained on 2.8M fp–structure pairs (test-scaffold-excluded), end-to-end finetune. De novo SOTA; retrieval-via-generation H@1 32.3/H@5 54.4.

**C. Generative / diffusion fingerprints (slot-fillers, not rankers).**
- **MIST+MolForge (2025):** MIST 4096-bit probs → threshold (t=0.5) on-bits → pretrained MolForge transformer → SMILES. CORRECTED numbers: 10.7% top-1 / 14.5% top-10 (MassSpecGym) — the viral 28%/36% was a **batch-size>1 attention-mask bug** (`attn += 0/1 mask` instead of −inf; padding leakage toward longest spectrum). Cautionary tale for our own PMA/masking code.
- **GLMR (Nov 2025):** contrastive pre-retrieval → generative LM conditioned on top candidates → unimodal re-rank; claims 64% H@1 — unverified under the 2026 audit; do not plan on it.
- **MassSpecGym-in-the-Wild audit (Jun 2026, → v1.5):** kills DreaMS+ChemBERTa (82→6.7%), ChemFormer-2stage (64→7.6%), SMILES-classifier (99.7→8.2%) via canonicalization fixes; flags PubChem-ranking bias (50% H@1 spectrum-blind!) and the MIST batch-mask bug. Use v1.5 canonicalization + spectrum-blind baselines in all our evals.

**Expected gains (honest, transfer to our LB scale).** Contrastive `f·z`/Emb-Cos retrain of same encoder: +0.01–0.03. True subformula labeling + MAGMA aux loss + forward (ICEBERG) augmentation (full MIST recipe): +0.02–0.05. Formula-conditioning (MIST-CF gate): +0.01–0.02 plus recall. FLARE-style peak↔atom head: +0.02–0.04 if it transfers. Generative slot-fillers: ~0 MRR, optional slots 16–25.

## 4. Why OUR three attempts underperformed

| | v2 MLP (`v2/train_fp.py`, README: blend 0.047 / learned 0.038) | v8 (`v8/train_fpt.py`, README: recipe that "scores 0.468 alone") | v-fpnet (`v-fpnet/train_fpnet.py`, `docs/fpnet.md`: pooled@5 0.599 / disj@5 0.548) |
|---|---|---|---|
| Loss | Pointwise **BCE** over 2048 bits, no decoys — optimizes per-bit calibration, not ranking (Pareto result: BCE is worst at retrieval) | Softmax CE over 63 in-window decoys ✓ | Same decoy CE ✓ (τ=10) |
| Negatives | None (blend = cosine-neighbor memory) | Same-mass window ✓ but truth-injection logic leaks easy negatives (`tgt=names.index(t) if in else 0` + `F[j,0]=truth` fallback) | Fresh same-mass decoys/epoch ✓, truth fixed pos 0 ✓ |
| Input | **12k 0.1-Da bins**, L2-norm — no chemistry; 0.21 val Tanimoto, 5 epochs, Adam 3e-4 | Same 12k bins (full-spectrum tail kept — why it beats fpnet) | **Top-80 set truncation** discards weak-peak tail; subformula signal is a cheap trick (28,939-mass library from top-150 formulae; per-peak hit+logppm only) vs MIST's per-peak formula embeddings + pairwise neutral losses + MAGMA aux loss |
| Architecture | 12k→1024→512→2048 MLP | 12k→1536→768→2048 MLP (still no peak interactions) | Set-transformer d=256/2 layers/4 heads + PMA + 31-dim covariates — right family, half their capacity (theirs d=512/6L/8H) |
| Ranking geometry | Sigmoid + Tanimoto/cosine | **L2-normalized cosine / τ=0.1** — contradicts the `f·z`-raw prescription; `v-fpnet/baseline_eval.py` shows geometry alone moves pooled@5 0.80→0.67 | Raw `f·z` ✓ |
| Checkpoint | Last-epoch (5 epochs) | Best-val retrieval ✓ | Pooled@5 ✓ (ep10/34) but pooled-truth-in-pool metric rewards memorization; loss 2.05→0.86 while retrieval flat from ep8 |
| Result | pooled@5 **0.269** | pooled@5 **0.670** / disj@5 0.630 (best of ours) | pooled@5 **0.599** / disj@5 0.548 |

**One-line diagnosis.** v2 failed on loss (BCE, no decoys) + input (bins) + undertraining; v8 fixed the loss but kept the wrong encoder and wrong geometry — yet full-spectrum bins beat our truncated set inputs; v-fpnet got loss+geometry+family right but lost on input fidelity (top-80 + impoverished subformula flags) and capacity, and validated on a memorization-prone metric. None of ours implements the actual MIST differentiator: per-peak **formula embeddings with pairwise neutral-loss modeling**, MAGMA aux supervision, or forward-model augmentation.

## 5. Upgrade plan — ranked by expected LB delta, MPS- or T4-feasible

1. **Re-rank with raw `f·z` everywhere; re-checkpoint v8/v-fpnet on disjoint@5 (+0.005–0.015, 0 GPU).** One-line eval change + `meta.json`-style best-disjoint selection. Fixes known geometry/metric bugs before any training.
2. **Contrastive retrain of the v8 binned MLP (Emb-Cos/InfoNCE + hard same-mass decoys, 150k spectra, ~2–4h T4 / overnight MPS) (+0.01–0.03).** Keep v8 featurization; replace cosine-τ loss with temperature-tuned contrastive `f·z`; fresh decoys/epoch; disjoint@5 checkpointing; ensemble 2–3 seeds. Cheapest retrieval-loss win.
3. **Hybrid input: full-spectrum bins + set-transformer peaks (modify `v-fpnet`, ~4–6h T4) (+0.015–0.035).** Concatenate binned-residual vector to the PMA-pooled vector before the 1024-MLP; lift truncation 80→256 with intensity-weighted pooling; keep raw `f·z`. Directly fixes the measured fpnet<v8 gap.
4. **True MIST input stack on the same encoder (~6–10h T4, possibly 2 sessions) (+0.02–0.05).** Per-peak subformula embeddings (RDBE-filtered ≤15 ppm enumerator already in `v1/subformula.py`), sinusoidal formula-count embeddings, pairwise neutral-loss terms, MAGMA/fragment-fingerprint aux loss (λ≈8), ICEBERG/ms-pred forward augmentation (precompute offline). This is the literature's whole gap (2.5→14.6).
5. **MIST-CF formula gate as pre-filter + GBM feature (~3–5h T4) (+0.01–0.02 + recall).** Energy-based formula ranker, ≤256 candidates; feed top-formula agreement + formula-mass residual into the ranker. Attacks failure mode 7 and the formula bottleneck (`docs/constraints-and-literature.md` constraint 4).
6. **FLARE-lite peak↔atom alignment head (research spike, T4) (+0.02–0.04 if transfers).** GNN atom encoder on candidates (precompute once), bidirectional peak-atom InfoNCE alongside `f·z`; inspectable alignments double as diagnostics.
7. **De novo slot-fillers only after 1–6 plateau (~0 MRR).** Pretrained MolForge/DiffMS decoder on thresholded predicted bits for slots 16–25; batch-size-1 inference (mask-bug lesson); spectrum-blind baseline required.

**Suggested order under quota:** 1 (today, CPU) → 2 (one T4 session) → 3 (one T4 session) → 5 (parallelizable on MPS) → 4 (the big train) → 6 (spike) → 7 (optional).
