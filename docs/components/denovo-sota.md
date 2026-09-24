# De novo SOTA for recall fills (slots 16–25): per-paper verdicts + ranked strategy

Branch context: retrieval pipeline already at ~0.61 local / 0.30+ public LB (`docs/components/`); survey finding 14 says generators score ~0 exact-match.
This doc scopes ONE job: lottery-ticket exact-match fills for slots 16–25 under exact-match MRR@25.
**Similarity (Tanimoto/MCES) does not score** — it is only useful as a rerank feature, never as an objective.

## 0. Paper inventory (`docs/papers/denovo/`, all `file`-verified PDFs unless noted)

| # | Paper | File | Source |
|---|-------|------|--------|
| 1 | MS2Mol + EnvedaDark + reranker/confidence (Enveda, 2023) | BLOCKED — ChemRxiv Cloudflare-gated (both v1 `648dead5…` and v4 `64925075…` asset URLs return 8 KB HTML stubs, deleted). Verdict below from full-text search extracts + SpecTUS/review cross-cites | doi:10.26434/chemrxiv-2023-vsmpx |
| 2 | MSNovelist (Nat Methods 2022) | `msnovelist_natmethods2022.pdf` (10 pp, 3.1 MB) | nature.com OA PDF |
| 3 | MassGenie (Biomolecules 2021) | `massgenie_biomolecules2021.pdf` (10 pp, 3.2 MB) | MDPI OA PDF |
| 4 | Spec2Mol (Commun Chem 2023) | `spec2mol_communchem2023.pdf` (12 pp, 1.1 MB) | nature.com OA PDF |
| 5 | Mass2SMILES (bioRxiv 2023.07.06.547963) | `mass2smiles_biorxiv2023.pdf` (27 pp w/ SI, 13 MB) | bioRxiv full PDF |
| 6 | DiffMS (arXiv 2502.09571) | `diffms_2502.09571.pdf` (20 pp, 1.3 MB; copy of `../structure/`) | arXiv |
| 7 | MIST + MolForge (AI4Mat-NeurIPS25, arXiv 2508.04180v4) | `mist-molforge_2508.04180.pdf` (8 pp) | arXiv |
| 8 | MolForge fp→SMILES (J Cheminf 2023) | BLOCKED — BMC/Springer bot-walled (PDF + HTML return 4 KB stubs, deleted). Verdict from GitHub README (`molforge_README.md`, on disk) + paper #7 | doi:10.1186/s13321-023-00693-0 |
| 9 | MS-BART (arXiv 2510.20615; copy of `../structure/`) | `msbart_2510.20615.pdf` (11 pp) | arXiv |
| 10 | BART-TTT test-time tuning (arXiv 2510.23746) | `bart-ttt_2510.23746.pdf` (22 pp) | arXiv |
| 11 | SpecTUS EI-MS BART (Anal Chem 2026; arXiv 2502.05114) | `spectus_2502.05114.pdf` (12 pp) | arXiv |
| 12 | MADGEN (arXiv 2501.01950) | `madgen_2501.01950.pdf` | arXiv |
| 13 | TeFT fragment-tree rerank (Commun Chem 2024) | `teft_communchem2024.pdf` (11 pp) | nature.com OA PDF |
| 14 | Review: Molecules 2026 e769 (taxonomy + leakage analysis) | `denovo-review_molecules2026.pdf` | MDPI OA PDF |
| — | MASSISTANT SELFIES EI-MS (J Chromatogr A 2025) | BLOCKED — ChemRxiv gated, Elsevier paywalled (S2 API has no OA copy). Verdict from abstract + ChemRxiv text extracts | doi:10.1016/j.chroma.2025.466216 |
| — | OMG RL-generation (Anal Chem 2025) | PAYWALLED (ACS). Verdict from abstract + `HassounLab/OMG` README | doi:10.1021/acs.analchem.5c01770 |
| — | MS2SMILES H-constrained LSTM (IEEE BIBM 2023) | PAYWALLED (IEEE). Verdict from abstract + review cross-cite | doi:10.1109/bibm58861.2023.10385903 |

All `txt/` extractions via `pdftotext -layout` in `docs/papers/denovo/txt/`.

## 1. Per-paper verdicts

### P1. MS2Mol + EnvedaDark + reranker + confidence (Butler/Enveda, ChemRxiv 2023, v4 Sep-2023)
- **Setup:** end-to-end spectrum→SMILES on *dark* NPs: EnvedaDark = 226 natural products absent from major DBs; plus CASMI-2022 and EnvedaLight (454 known NPs). Metrics are Tanimoto thresholds calibrated to blinded chemist usefulness votes (meaningful-sim / close-match), NOT exact match.
- **Architecture:** BART-style encoder-decoder; innovations: (a) BPE SMILES vocab (learned substructure tokens), (b) precursor mass input with random masking during training, (c) fragment m/z as integer+fractional token pairs (keeps HRMS resolution, small vocab), (d) **XGBoost NDCG reranker** over beam candidates trained on a structure-disjoint held-out split, (e) **GBM confidence regressor** predicting Tanimoto to truth from spectrum/prediction/log-prob/mass-diff features.
- **Training data:** ~1 M spectra (public libraries + internal Enveda; exact split undisclosed).
- **Metrics:** EnvedaDark 21% close-match / 62% meaningful-sim (+95%/+44% over DB baselines); top-10%-confident subset 63.4% / 98% (MAE 0.13, R² 0.40). Exact-match rate not reported (effectively ~0 on dark matter).
- **Code/weights/license:** PROMISED "at publication", NEVER RELEASED — `github.com/enveda` org (37 repos) has no MS2Mol repo as of 2026. No weights, no license. **Unrunnable.**
- **Offline verdict:** NO-GO as a runner. GO as a design donor: BPE-SMILES, precursor-masking, int/frac m/z tokens, and especially the XGB-rerank + confidence-regressor pattern (port directly to our fills, §3-R4).

### P2. MSNovelist (Stravs et al., Nat Methods 2022)
- **Setup:** two-stage: SIRIUS formula + CSI:FingerID 3,609-bit probabilistic fingerprint → LSTM encoder-decoder writes SMILES under formula constraint → top-128 → re-rank by modified Platt score (fingerprint match).
- **Architecture:** 3-layer encoder → latent z → 3-layer LSTM decoder, char-level SMILES + running subformula state; beam search.
- **Training data:** fingerprint→structure decoder trained on 1,232,184 structures (HMDB+COCONUT+DSSTox, eval structures removed); needs ZERO paired spectra (key trick).
- **Metrics:** GNPS 3,863 spectra: 25% top-1 / 45% retrieved (top-128); CASMI-2016: 26% / 57%. Authors' own naïve-generator control shows GNPS numbers are leakage-inflated (review P14 agrees: no MCES split). MassSpecGym (DiffMS re-impl, formula-aware): 0.00% exact.
- **Code/weights/license:** data on Zenodo/GMTLS; generator code "not readily retrainable" (per DiffMS authors); production path is via SIRIUS (Java, fragmentation-tree bottleneck, academic-free). No clean pip/weights artifact.
- **Offline verdict:** NO-GO for the kernel (SIRIUS per-spectrum cost + no retrainable checkpoint). Concept validated by P7 instead: fp→structure decoding works, fingerprints are the right intermediate.

### P3. MassGenie (Shrivastava/Kell, Biomolecules 2021)
- **Setup:** spectrum→SMILES cast as translation; trains on **in-silico** fragments (FragGenie: MetFrag-style recursive bond-breaking, depth 3, single non-aromatic bonds, tuned vs MoNA) + fine-tune on experimental; inference sweeps top-k peak subsets × 300 samples, formula-filters, then "round-trips" (re-fragment candidates with FragGenie, cosine-rank) + VAE-Sim neighbor expansion.
- **Architecture:** vanilla transformer, ~400 M params, augmented-SMILES targets, 19 h train.
- **Training data:** ~6 M structures (paper) / 4.7 M synthetic + ~200 k GNPS experimental (per SpecTUS cite); MS ≤ 500 Da, +ESI only.
- **Metrics:** 49/93 (53%) exact on CASMI-2017-era set — pre-MCES-split era, treat as inflated; strong on in-silico spectra, degrades on experimental ("sim-to-real gap", per P14).
- **Code/weights/license:** FragGenie on GitHub (`neilswainston/FragGenie`); transformer weights "will be made available" — never were. No license artifact.
- **Offline verdict:** NO-GO as runner. Two portable ideas: (a) in-silico pretraining at million-scale (validates P6/P9 strategy), (b) round-trip ranking (re-fragment candidate, cosine vs query) as a cheap physics-based fill-reranker — implementable with our frag rules + CFM-style scorer.

### P4. Spec2Mol (Litsa et al., Commun Chem 2023)
- **Setup:** GRU SMILES autoencoder pretrained on 138 M structures → freeze decoder → 1-D CNN spectrum encoder trained to hit the same latent space (28 k NIST spectra); direct sampling + indirect (nearest-neighbor in latent space) generation; MW re-rank top-20.
- **Training data:** NIST20 experimental; evaluated where SIRIUS formula FAILS (adversarial slice for CSI:FingerID).
- **Metrics:** similarity-only (fingerprint cosine, MCS ratio/coef ≈ 0.6–0.74 max over candidates); ~0 exact (MassSpecGym re-train: 0.00%).
- **Code/weights/license:** paper's code NOT public ("codes to Spec2mol … are not available", per P5). Unrunnable.
- **Offline verdict:** NO-GO. Lesson: latent-space bridging without formula constraint gives analogs, not exact hits — wrong objective for MRR fills.

### P5. Mass2SMILES (Elser/Huber, bioRxiv 2023; NOTE: distinct from IEEE MS2SMILES below)
- **Setup:** 5-layer transformer encoder → TCN → continuous molecular-descriptor space (from a molecule-design VAE) + 60 functional-group + formula + adduct heads; nearest SMILES decode.
- **Training data:** 83,358 GNPS + NIST-HRMS natural-product spectra + computed neutral losses; 744-spectrum InChIKey-held-out validation.
- **Metrics:** 7/744 (~1%) exact, 14 with Tanimoto > 0.9, mean Tanimoto 0.39 (valid) / 0.40 (CASMI-2022, 2 exact). Honest low numbers.
- **Code/weights/license:** `github.com/volvox292/mass2smiles`, Docker CPU inference ~2 s/spectrum; NO license declared; preprint CC-BY-NC.
- **Offline verdict:** RUNNABLE but WEAK — 33 M params, CPU-cheap, yet ~1% exact-match makes it a worse lottery ticket than P7. Only value: functional-group head as auxiliary rerank features. Rank: fallback.

### P6. DiffMS (Bohde/Coley, arXiv 2502.09571)
- **Setup:** formula-conditioned discrete graph diffusion (DiGress-style): MIST spectrum encoder (formula transformer over annotated peaks) → precursor-peak embedding conditions graph-diffusion decoder over adjacency matrix; frequency-based ranking of samples.
- **Training data:** 2.8 M fingerprint–molecule pairs (DSSTox/HMDB/COCONUT/MOSES, test/val removed) for decoder pretrain; NPLIB1/MassSpecGym fine-tune.
- **Metrics:** NPLIB1 8.34% top-1 / 15.44% top-10; MassSpecGym 2.30% / 4.25% (top-10 Tanimoto 0.39). Best pre-2025 leakage-controlled result (P14's "4.1% SOTA" = this row).
- **Code/weights/license:** `github.com/coleygroup/DiffMS`, **MIT**; MIST checkpoints on Zenodo `15122968`. Fully open.
- **Offline verdict:** RUNNABLE, rank #2 fill source. Cost: diffusion sampling (hundreds of steps × graph transformer) is the heaviest CPU option here; viable only for a small molecule subset or with reduced steps. Formula-conditioning is mandatory (it is the main accuracy driver).

### P7. MIST + MolForge (Neo et al., AI4Mat-NeurIPS25, arXiv 2508.04180v4) — READ FULLY
- **Setup:** MassSpecGym de-novo WITH formula: pretrained MIST (MassSpecGym) → 4096-bit Morgan fingerprint probs → threshold → on-bit indices → MolForge autoregressive transformer → beam-10 SMILES.
- **Architecture:** MIST = chemical-formula transformer per peak + fingerprint head; MolForge = encoder-decoder transformer, src = on-bit indices, tgt = SMILES tokens (SentencePiece). Key tricks: (a) MolForge retrained on ~2.8–3 M DiffMS compounds (17 k MassSpecGym alone → 0%!), (b) **prior-adjusted threshold** t=0.172 (match train on-bit rate 1.09%) beats t=0.5.
- **Metrics (REPORTED v4):** 31.0% top-1 / 40.0% top-10, MCES 12.4, Tanimoto 0.68. **⚠️ CORRECTED (third-party repro `harrylaucngd/MIST-MolForge`): 10.73% top-1 / 14.48% top-10, MIST Tanimoto 0.457.** The v4 numbers are inflated by a MIST padding-mask bug (`attn += attn_mask` with 0/1 instead of −inf) that leaks across batched samples at batch-size 24; batch-size-1 inference removes it. Ground-truth-fingerprint ceiling: 46%/59% — the ENCODER is the bottleneck, decoder is excellent.
- **Code/weights/license:** MIST MIT (checkpoints Zenodo `15122968`: `mist_msg.pt`); MolForge repo CC **BY-NC 4.0** badge (license file NOASSERTION via API — treat as NC), checkpoints on Google Drive + OSF mirror via repro repo; repro repo is self-contained with fix. Retraining MolForge took 3 days on 1×A40 (we will NOT retrain — reuse checkpoints).
- **Offline verdict:** RUNNABLE, rank #1 fill source (corrected ~11% top-1 still >2× DiffMS). Caveats: (a) MUST run MIST at batch-size 1 or with the mask fix; (b) NC-licensed decoder weights — flag for competition use, prefer OSF/repro artifacts, keep retrieval as the scored backbone so fills are severable; (c) needs subformula-annotated peaks (our v1 subformula labeller covers this).

### P8. MolForge fp→structure (Ucak et al., J Cheminf 2023) — via README + P7
- **Setup:** pure fingerprint→SMILES/SELFIES translation over 13 fingerprint types; ECFP4→SMILES 92.1% "Tanimoto exactness" WITH TRUE fingerprints (decoder ceiling, not a spectra result).
- **Code/weights/license:** `github.com/knu-lcbc/MolForge`, checkpoints via Google Drive, CC BY-NC 4.0 badge.
- **Offline verdict:** component of P7; standalone use = decode OUR FPNet fingerprints (6,930-bit ECFP4/6+RDKit+MACCS — mismatched to MolForge's 4096-bit Morgan-R2, would need retrain → skip; use P7's MIST pairing as published).

### P9. MS-BART (arXiv 2510.20615; OpenDFM) — MS/MS, SELFIES, BART
- **Setup:** MIST-thresholded fingerprints + SELFIES in ONE vocabulary; 4-task pretraining (SELFIES denoise, fp→mol translation, hybrid both-orders) on 4 M pairs (MassSpecGym mols, MCES≥2 from test) → finetune on experimental → Tanimoto-preference alignment (decoder-only).
- **Metrics:** NPLIB1 7.45%/10.99% (Tanimoto 0.44/0.51, best similarity); MassSpecGym 1.07%/1.11% (strict MCES≥2 pretrain filter — authors show DiffMS's higher number partly reflects looser filtering). Gold-fingerprint ceiling: 47.6%/64.6% — again, encoder bottleneck. 10× faster inference than diffusion.
- **Code/weights/license:** `github.com/OpenDFM/MS-BART`, NO license declared; weights presumably with release (verify before kernel use).
- **Offline verdict:** RUNNABLE-IF-WEIGHTS, rank #3. SELFIES output = 100% valid strings (matters for fill-slot efficiency: no wasted invalid SMILES). Best when Tanimoto-shaped candidates are wanted (analog channel), but for exact fills P7 dominates.

### P10. BART-TTT (arXiv 2510.23746) — formula-constrained BART + test-time tuning
- **Setup:** `facebook/bart-base` seq2seq on spectrum+formula text → SMILES + auxiliary fingerprint head; **formula-constrained decoding** (ban tokens violating formula); optional per-query test-time gradient tuning.
- **Metrics:** NPLIB1 12.88% top-1; MassSpecGym 3.16% top-1 (fine-tune; TTT adds +62% relative on MassSpecGym in their setup). Ablations that matter for us: Na⁺-adduct spectra 0.09% top-1 (vs 3.9% H⁺), >600 Da ~0%, QTOF > Orbitrap.
- **Code/weights/license:** `github.com/rxn4chemistry/MultimodalAnalytical/tree/ttt-msms` (+ pretrained sim model). License TBD (verify).
- **Offline verdict:** NO TTT in kernel (per-query gradient steps too costly); PLAIN fine-tuned checkpoint COULD serve as rank-#3 alternative to P9 if weights are posted. Portable idea regardless: formula-constrained decoding — apply as a POST-FILTER on any generator's outputs (cheap, no model needed).

### P11. SpecTUS (Hájek et al., Anal Chem 2026; BART-354M, **EI-MS only**)
- **Setup:** GC-EI-MS (integer m/z + binned intensity embeddings; positional channel reused for intensity) → char-SMILES; 2×8.6 M NEIMS+RASSP synthetic pretrain (58 h H100) → 232 k NIST20 finetune; per-candidate certainty score.
- **Metrics:** NIST-held-out 43% top-1 / 65% top-10 exact; 76%/84% beat hybrid DB search (similarity). EI-MS only — NOT transferable to ESI-MS/MS (no precursor, different fragmentation).
- **Code/weights/license:** `github.com/hejjack/SpecTUS` **MIT**; synthetic-data training scripts + tutorial open; FINETUNED weights need NIST20 license; synthetic-pretrained (`MS-ML/SpecTUS_pretrained_only`, HF) is open but near-useless on real spectra (3% NIST recon pre-finetune).
- **Offline verdict:** NOT APPLICABLE to this competition (EI vs ESI-MS/MS). Value = CPU calibration: 354M-param BART does 10 candidates in **36 s on a Xeon Gold 6130 CPU** (8 s single) — proves CPU inference of P7/P9-class models fits a 9 h kernel (§4 math).

### P12. MADGEN (HassounLab, arXiv 2501.01950) — scaffold-conditioned
- **Setup:** Stage-1 contrastive scaffold retrieval (spectrum↔Murcko scaffold) → Stage-2 graph-transformer (RetroBridge) completes molecule from scaffold + spectrum + formula.
- **Metrics:** MassSpecGym predictive 1.31%/1.54% vs ORACLE-scaffold 10.5%/12.4%; NIST23 oracle 49.0%/65.5%. Scaffold prediction (13–40% SPA) is THE bottleneck.
- **Code/weights/license:** `github.com/HassounLab/MADGEN` (license TBD).
- **Offline verdict:** NO-GO for fills (predictive path < DiffMS; oracle path inapplicable). Portable idea: scaffold-constrained generation for hypothesis-driven slots (e.g., force generation around a high-confidence retrieval scaffold) — future work, not v1 fills.

### P13. TeFT (Yang et al., Commun Chem 2024) — transformer + fragment-tree rerank
- **Setup:** 65 M-param transformer proposes SMILES list (100 stochastic runs) → RECAP/SMARTS simulated-fragmentation "SMILES trees" → align vs SIRIUS-style fragmentation tree from spectrum → similarity score ranks + annotates peaks. Built for LOW-RES miniaturized MSn.
- **Metrics:** GNPS/HMDB/MoNA/CASMI-2017: top-N similarity wins, exact-match below SIRIUS4 ≈ MetFrag (CASMI-2017 top-1 27.9% vs 26.7% MetFrag — library-era numbers, leaky).
- **Code/weights/license:** `github.com/thumingo/TeFT`, no license declared.
- **Offline verdict:** NO-GO as generator. Portable idea: **fragment-tree alignment as a physics-based fill-reranker** (same family as P3 round-tripping) — score P7's beam-10 per candidate with our own subformula-tree match instead of raw log-prob.

### P14. Review (Schneider et al., Molecules 2026) — field taxonomy, read for calibration
- Three eras: fp-conditioned RNN → end-to-end seq → formula-conditioned graph diffusion; "SOTA 4.1% top-10" = pre-MolForge-correction DiffMS row (their cutoff predates P7v4; use P7-corrected 14.5% as current best).
- Core findings we adopt: (1) GNPS-split numbers (MSNovelist 45%, MS2SMILES 45%) are leakage-inflated — trust only MCES-split (MassSpecGym) numbers; (2) oracle formula conditioning adds several points for every architecture; predicted-vs-oracle gap is small for DiffMS (4.25→4.10) but catastrophic for scaffold models; (3) generators are hypothesis engines, not identifiers — matches our slots-16–25 design.

### Unobtained in full (blocked/paywalled — verdicts from abstracts, code READMEs, P14)
- **MASSISTANT (SELFIES, EI-MS, J Chromatogr A 2025):** MLP spectrum→SELFIES, ≤600 Da NIST08; ~10% exact full-NIST, 54% on curated homogeneous subset (dataset-curation effect, mirrors P11). EI-MS + thin architecture detail → NOT APPLICABLE; SELFIES-validity lesson already covered by P9.
- **OMG (REINVENT4 TL+RL + JESTR/ESP rank, Anal Chem 2025):** CANOPUS 10.51% / MassSpecGym 2.42% top-1 — best published MassSpecGym top-1 among non-P7 methods. BUT: per-query finetuning + RL sampling per spectrum = orders of magnitude too slow for a 9 h CPU kernel. NO-GO as runner. JESTR (`HassounLab/JESTR1`, MIT, `../structure/jestr_2411.14464.pdf` on disk) as a *ranker* of generated candidates is the portable piece — aligns with our GBM-rerank plans.
- **MS2SMILES (IEEE BIBM 2023, Kell group):** H-aware SMILES + grammar-constrained LSTM decode; 53.6% GNPS / 63.8% CASMI-2016 "SMILES accuracy" (leaky splits per P14 — discount); code inside `idslme/IDSL_MINT` repo. NO-GO (paywalled, leaky eval, LSTM < transformer options above).

## 2. Cross-paper facts that bind our design

1. **Exact-match is ~0 for all pure generators on leakage-controlled data EXCEPT the MIST-family:** MassSpecGym top-1 = DiffMS 2.3%, MADGEN-pred 1.3%, MS-BART 1.1%, BART-TTT 3.2%, OMG 2.4%, MIST+MolForge-corrected **10.7%**. Expect 1–5% on our timsTOF-NP test (domain shift down from MassSpecGym) — i.e., a handful of exact hits over hundreds of molecules, each worth 0.04–0.06 MRR sitting at ranks 16–25. Positive-EV lottery tickets; never displace retrieval ranks 1–15.
2. **The encoder is the bottleneck everywhere** (gold-fingerprint ceilings 46–65% vs 1–11% realized). Our FPNet channel (0.468 MRR alone) is already a strong encoder — a future MIST↔FPNet swap or FPNet→MolForge experiment is the highest-upside follow-up (needs bit-space retraining; NOT v1).
3. **Formula conditioning is free accuracy.** Every architecture gains; enforce MIST-CF-formula agreement as a hard fill filter (§3).
4. **SMILES invalidity wastes beam slots; SELFIES doesn't** (P9/P11). Prefer SELFIES-output or validity-filtered beams for fills.
5. **Adduct/instrument shift is brutal** (P10: Na⁺ 0.09% vs H⁺ 3.9%; P11: EI≠ESI). Gate fill generation on H⁺/common adducts + QTOF/timsTOF-like spectra; skip exotic adducts.
6. **Confidence/rerank transfers better than generators** (P1 XGB-rerank/confidence, P3 round-trip, P13 tree-alignment, OMG JESTR/ESP). Our GBM ranker absorbs these as features.

## 3. Ranked fill-strategy recommendations (slots 16–25 only)

- **R1 (implement): MIST (MIT, Zenodo) + MolForge (NC-note) with corrected inference.** Batch-size-1 MIST (or mask fix), prior-adjusted threshold t≈0.17, beam-10 decode, keep top ≤10 novel (non-pool, formula-agreeing, inchikey14-deduped) structures per molecule for slots 16–25. Expected: best exact-hit rate available offline.
- **R2 (implement): hard physics filters before any fill is seated.** (a) MIST-CF formula agreement (exact mass + formula match), (b) precursor-mass agreement ≤10 ppm, (c) validity + dedup vs pool and vs ranks 1–15, (d) adduct gate (H⁺/Na⁺-common only). A mass-violating fill is a burned slot.
- **R3 (implement): MS2Mol-style confidence/XGB rerank over fills.** Features: beam log-prob, MIST bit-mean-prob, formula-agreement flags, mass-diff, retrieval-rank agreement (does any fill match a rank 5–25 candidate? — upgrade it), round-trip FragGenie-style fragment cosine (P3-port). Order fills by this score, retrieval stays 1–15.
- **R4 (if time): DiffMS-pretrained diffusion (MIT) as second-chance generator** on molecules where R1 yields nothing valid; reduced diffusion steps for CPU. Strictly additive to R1.
- **R5 (explicit non-goals):** MS2Mol weights (don't exist), retraining MolForge/DiffMS (GPU-weeks; A40×3d reference), TTT/RL per-query adaptation (P10/BART-TTT, OMG — too slow), EI-MS models (P11/MASSISTANT — wrong modality), SIRIUS-dependent paths (P2 — too slow), scaffold-oracle paths (P12 — inapplicable).

## 4. What it takes to run R1 offline in a 9 h CPU kernel (checklist)

1. **Vendor NOW (online):** `mist_msg.pt` (+ `mist_canopus.pt` optional) from Zenodo 15122968; MolForge checkpoint + SentencePiece model via repro-repo OSF mirror (`harrylaucngd/MIST-MolForge`); MIST `main_v2` + MolForge source trees; RDKit. Sizes are 100s-of-MB total — fits kernel bundles. No internet at scoring time.
2. **Peak annotation:** MIST needs subformula-annotated peaks — reuse the v1 subformula labeller (`constraints-and-literature.md` Immediate-v1 #1); fallback to unannotated incurs Tanimoto 0.73→0.63 (P7 §4.2) — acceptable degradation, keep fallback path.
3. **Inference config:** batch-size 1 (the bug fix), fp-threshold 0.172 (recompute on OUR train on-bit rate), beam-10. CPU calibration from P11 (354M BART: 36 s/10 cands on Xeon-6130): MIST encoder + small MolForge decoder ≈ 30–60 s/molecule → ~500–1000 molecules per 9 h single-worker; batch molecules across workers / restrict to H⁺-adduct, pool-miss molecules first.
4. **Slot discipline:** fills enter ONLY at 16–25, after inchikey14 dedup vs ranks 1–15 and vs pool top-25; cap 10 fills/molecule; log per-molecule provenance (generator, threshold, formula-check) for ablation.
5. **Validation:** measure on structure-disjoint split: fill exact-hit rate + ΔMRR when appended at 16–25. Ship-kill criterion: if fills add <0.002 local MRR, drop them (they cost kernel minutes and NC-license surface).
6. **License hygiene:** MIST MIT ✓; MolForge CC BY-NC 4.0 (badge; API NOASSERTION) — keeps fills severable: kernel runs and scores identically with `--no-denovo` if NC terms are ever an issue; retrieval backbone untouched.

## 5. Bottom line

No generator touches retrieval for ranks 1–15 (best leakage-controlled top-1 is 10.7% vs our 0.30+ LB retrieval). But MIST+MolForge-corrected is a legitimate exact-match lottery for dark-matter slots: ~1–5% expected hit rate × ~0.05 MRR per hit = visible LB delta for ~1 h of CPU. Implement R1–R3 with R2 hard filters; everything else is a design donor, not a runner.
