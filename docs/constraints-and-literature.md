# Constraints + Literature: what binds winning, and what the papers say
Branch: feat/domain-research | Date: 2026-09-21 | Scoreboard: v0 0.088 vs top 0.409

## Papers on disk (`docs/papers/`)
1. `massspecgym-neurips24.pdf` - MassSpecGym, NeurIPS 2024 Spotlight (231K spectra / 29K structures, MCES-split benchmark). READ.
2. `mist-cf-arxiv.pdf` - MIST-CF, JCIM 2024 (formula ranking, energy-based, tied CASMI2022 winner). READ.
3. `mist-biorxiv.pdf` - MIST, Nat Mach Intell 2023 (formula transformers, fingerprint retrieval). DOWNLOADED, key results via MassSpecGym + GitHub (29MB, read on demand).
4. MS2Mol (Enveda, ChemRxiv 2023) - PDF blocked by Cloudflare; full text via search snippets + author overlap with this competition.

## The constraint stack (user's question, ordered by binding force)
1. **Candidate recall:** truth must be inside your 25. Test = MCES-novel natural products; PubChem-scale DBs can't be enumerated per molecule. MassSpecGym caps retrieval at |C|<=256 same-mass candidates - same funnel we need. Miss = 0, no recovery.
2. **Rank-1 precision:** MRR pays 1.0 / 0.5 / 0.33... rank25 = 0.04. Everything past rank ~5 is nearly worthless. Optimise top-1, not recall@25 alone.
3. **Generalization, not memorisation:** MassSpecGym proves 2D-InChIKey splits leak (Tanimoto>0.85 near-dupes across folds); they demand MCES>=10 separation. Our validation MUST be structure-disjoint (inchikey14 at minimum) or LB will punish us. Train is 99% out-of-domain (synthetic/Orbitrap) vs test (timsTOF natural products).
4. **Formula bottleneck:** CASMI median formula accuracy 71%, max 94%. Formula constrains candidates to isomers (orders of magnitude pruning). MIST-CF: 0.769 top-1 (NPLIB1+NIST20), tied SIRIUS CASMI2022 submission 0.868 joint with zero curation at 1/3 runtime. Get formula right first; errors cascade to structure.
5. **Compute:** 9h offline GPU/CPU, no internet. MIST-CF trains <3h on 1x RTX A5000; top-20 peaks saturate accuracy (diminishing past 20, Np=50 marginal). Design for top-20 peaks, precomputed FAISS, quantised.
6. **Slot economy:** 25 slots only. Tautomer/stereo duplicates waste them. Dedup by inchikey14 is mandatory (public notebooks already do "Tautomer Dedup").
7. **Feedback budget:** 5 subs/day, 33/67 public/private split. Every submit must test a hypothesis; local structure-disjoint CV is the real LB.

## What the numbers say (MassSpecGym retrieval, no formula)
- Random: hit@1 0.37 / @5 2.01 / @20 8.22
- Fingerprint FFN: 2.54 / 7.59 / 20.00
- DeepSets+Fourier: 5.24 / 12.58 / 28.21
- MIST: 14.64 / 34.87 / 59.15, MCES@1 15.37
- Our v0 cosine ~= Random-DeepSets tier. The gap MIST opens comes from ONE thing: subformula-labelled peaks (domain encoding) instead of raw m/z bins. That is v1.
- De novo top-1 accuracy: 0.00 for ALL baselines (random, SMILES-TF, SELFIES-TF). Exact generation is unsolved - retrieval + rerank is the game; generative (MS2Mol-style) only as slot-fillers for dark matter.

## Design rules extracted
- Encode peaks as subformulae (RDBE-filtered, <=15ppm, top-20 by intensity), not binned m/z. (MIST/MIST-CF)
- Condition on instrument + adduct + collision_energy_ev as covariates. Rare adducts (K+) underperform in data-driven models - backstop with rules. (MIST-CF Fig 3e)
- Rank with energy-based softmax over true + hard decoys (same-mass isomers), not pointwise regression. (MIST-CF Eq 4)
- FastFilter pattern: cheap formula prior -> top-256 -> expensive scorer. 99% recall preserved. Copy for structures.
- Multi-spectra fusion per molecule (our 1-9 spectra/molecule): attention/mean over spectra, weight by base_peak_intensity. Not in papers (they're per-spectrum) - our edge.
- Semi-supervised: GeMS-A10 24M unlabeled spectra + 1M bio molecules exist (MassSpecGym). If v1 plateaus, contrastive pretraining is the next lever.
- MS2Mol (Enveda's own): BPE-SMILES BART, precursor masking, integer+fractional mass tokens, reranker + confidence scorer; EnvedaDark 21% close-match / 62% meaningful-sim, +95% over DB baselines. Implication: hosts believe generative + confidence wins dark space. Our 25 slots allow hybrid: retrieval ranks 1-15, generative/confident-novel fills 16-25.

## Immediate v1 (from papers, not guesses)
1. Subformula labeller (NumPy, RDBE>=0, 15ppm, top-20 peaks) - MIST-CF open routine.
2. Formula ranker (energy-based, decoys) -> isomer candidate sets per molecule.
3. Fingerprint predictor (Formula Transformer or FFN on subformula sets) -> cosine vs candidate fingerprints from train 277K + COCONUT.
4. Per-molecule fusion + LightGBM rerank (neural score, mass ppm, Tanimoto, neutral-loss count, library prior).
5. inchikey14 dedup, fill to 25. Validate MCES/inchikey-disjoint. Target: 0.20+ public.
