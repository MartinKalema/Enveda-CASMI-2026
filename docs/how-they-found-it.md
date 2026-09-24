# How they found it (from their own notebook narrative + paper trail)

## What they read
Papers (cited in-notebook): Li et al. 2021 spectral entropy; Dührkop/Böcker
CSI:FingerID PNAS 2015 (the f·z identity); GNPS molecular networking
(Watrous'12 modified cosine, Wang'16, Bittremieux'22); MIST (Goldman) +
MassSpecGym (Bushuiev) lineage; DreaMS foundation model (tested, rejected
with mechanism); CFM-ID/SIRIUS as reference points. Data: COCONUT download
page. Community (not papers): dariushafshar's ppm-window audit (121,805
spectra mis-indexed >10ppm), denpugovkin's adduct-grouping thread,
discussions 742055/742304, plus fork network (megayak adduct deltas, beraterolelk
tuning, dariushafshar formula-mass index).

## How the combination was discovered (their ablation ladder, all measured)
1. Library search alone: 0.151 (Class-1 banked).
2. +analog propagation: 0.233. The CORE idea came from GNPS networking:
   congeners share scaffold fragments at constant mass shift; validated on
   250 held-out NPs (0.164 mass-order → 0.521) AND 569 obscure structures
   (0.516) to rule out well-studied-compound artefact.
3. +class weights (LB-solved, not class share): 0.245.
4. +MetFrag-lite: 0.266. Independent evidence (corr 0.058).
5. +fingerprint model (f·z, hard decoys, best checkpoint): 0.299, then 0.335
   with longer retrain + merged/per-spectrum averaging (+0.019, their largest
   single LB effect).
6. Every constant via single-variable A/B with 1 sub each; regressions kept
   in the log (cap bug 0.282, PubChem 0.205, bundled-constants 0.311).
7. Noise discipline throughout: same code reruns 0.335→0.328; GBM default
   random_state=None cost 0.292 vs 0.298; ship threshold |Δ|≥0.012.

## Method (steal the process, not just the code)
- One hypothesis per submission; never bundles (their 0.311 regression
  autopsy proves why).
- Local validation for direction, LB for truth; distrust local when it
  disagrees twice in a row (their words: validation stops predicting).
- Dead ends logged with numbers (PubChem, consensus fp, confidence gate,
  same-view ensembles) so nobody retries them blind.
- Community as sensor: two of their biggest fixes came from others' threads.

## Grid-search plan (agent running)
Local grid over OUR upgrade combos on calibrated multi-seed validation
(grouped-by-molecule, anchor-gated, |Δ|≥0.012 to ship): analog K/p/window,
MetFrag on/off, fp-channel variants, GBM feature sets, dedup variants,
floor variants. Output: ranked combo table → v14 config. Only the winner
touches a submission.
