# Kaggle public-notebook survey (read-only, 2026-09-22)

Competition: `enveda-CASMI26-molecule-id-mass-spectra` (MRR@25, exact InChIKey14 after tautomer canonicalisation).
Our LB: 0.088 (mass-filter+cosine). Public top: ~0.34. All notebooks pulled via
`kaggle kernels list --competition ... --sort-by voteCount`, code read, nothing submitted.

## The one family that matters

Nearly all top notebooks are forks of one pipeline: **prvsiyan "Analog Propagation"** (95 votes,
living doc, LB 0.335) and its highest-scoring forks **haideptry "Fast Spectral Cosine Baseline"**
(183 votes, now same code, LB 0.339) / **"4-Channel Transformer Analog Ensemble"** (0.339),
**beraterolelk "Analog Ranker"** (0.336), **malikhammadfarooq "Quad-Channel"**, **flexonafft
"Strong Retrieval"**, **denpugovkin "Protected Bio-DB Tail"**. Treat them as one engine with
per-fork deltas. Core engine = 4 evidence channels + GBM ranker over a 711k pool:

- Pool = train structures (275,810 keys) ∪ COCONUT 2.0 (462,028 keys), deduped on InChIKey14,
  sorted by exact mass; fingerprint per structure `ECFP4+ECFP6+RDKitFP+MACCS` filtered to 6,930
  informative bits. Candidates = ±10 ppm neutral-mass window (median 56 candidates).
- Ch1 library search (entropy similarity, Li et al. 2021). Ch2 mass-shifted analog propagation.
  Ch3 MetFrag-lite bond-breaking. Ch4 spectrum→fingerprint transformer scored by `f·z`.
- Per molecule: median neutral mass over its spectra; lib/analog sims = max over spectra;
  model logits = mean of per-spectrum view and merged-peaklist view. GBM (HistGradientBoosting)
  maps 25–31 features to P(answer); seed-bagged (4 seeds × 2 class priors).

## Ranked findings — what to steal

1. **Analog propagation is the whole game for Class 2 (steal first).** Score candidates by
   `max_a sim(a)^p · Tanimoto(f_c, f_a)` over mass-shifted analogs (|ΔM|≤200 Da, keep top
   80–200, p=3–4). Measured: Class-2 MRR 0.16 (mass-error order) → 0.52. Needs no Class-1
   branch (self-match at ΔM=0 covers it). This is a GNPS molecular-networking idea reused as a
   ranking prior. Our pipeline has nothing like it — biggest expected gain.
2. **Tight window + NO candidate cap.** ±10 ppm beats ±20/±30 (0.521/0.509/0.500). Capping to
   top-80 by `lib_sim·100 − |Δmass|` discards 25% of reachable Class-2 answers (retention
   0.992→0.752) because Class-2 rows have lib_sim=0 by definition. Never pre-filter by mass
   proximity inside an already mass-selected window.
3. **PubChem expansion hurts; COCONUT-only is deliberate.** +PubChem isomers: 0.52→0.35, and
   the fingerprint channel does not rescue it (0.73→0.38). Reason: analog-Tanimoto rewards
   near-duplicates of analogs; truth is usually a derivative (median Tanimoto 0.83 to best
   analog). Implication for our pending COCONUT work: add COCONUT, do NOT add PubChem isomer
   lists; recall must come without near-duplicate decoys.
4. **Entropy similarity > cosine.** Entropy weighting of low-entropy spectra + intensity power
   1.0: Class-1 0.893→0.93. Cheap drop-in for our cosine scorer (numba kernels published).
5. **Fix our fingerprint approach: rank by `f·z`, train with in-window hard decoys.**
   Bayes LL over bits collapses exactly to dot product with raw logits (no calibration).
   Train with 63 decoys sampled from the *same ±10 ppm window* under softmax CE; keep
   best-by-validation checkpoint (models memorise: top-1 0.457@12k → 0.127@69k while loss
   improves); augment (peak dropout, intensity jitter, ±5 ppm m/z noise). Alone 0.468,
   +analog 0.567, all-four 0.612. Our fingerprint blend failed likely for lack of hard
   in-window negatives and checkpoint selection.
6. **MetFrag-lite validates our explained-intensity direction, but bond-breaking > formulas.**
   Break every 1–2 bonds, ±2 H rearrangements, score sqrt-intensity fraction explained
   (tol 0.01 Da). Alone 0.259; correlation with analog channel only 0.058, so it stacks
   (+analog → 0.545). Consider replacing our subformula enumeration with 1–2-bond cleavage.
7. **Calibrate, don't hand-weight; solve the class weight from the LB.** Adding lib-sim with
   fixed weight drops Class-2 0.52→0.27. GBM on ~25 features (lib block, analog block incl.
   max/mean/top-tan, model block raw/z/rank, frag block, log n_cand), sample_weight W1≈0.42
   solved from LB readings (not the 0.16 raw class share). LB noise floor ±0.006 from GBM
   seeds — bag seeds, distrust deltas <0.007.
8. **Index on formula mass or hybrid tolerance.** dariushafshar: 121,805 spectra disagree
   >10 ppm between formula-derived and precursor-derived neutral mass. riken = 0.005 Da
   precision issue (relative window fails at light masses); gnps = discrete adduct-label
   errors (e.g. +21.982 Da Na↔H). Fix: `max(10 ppm, 0.01 Da)` or formula-mass index
   (Class-1 0.9206→0.9253, narrower). Also: never flat-filter train at 10 ppm (loses 53%
   riken = plant metabolites closest to test chemistry).
9. **Metric-exact dedup (tautomer-canonical InChIKey14).** ~10% of COCONUT keys change under
   the scorer's canonicalisation; naive keying wastes top-25 slots and undercounts
   validation. Offline RDKit 2026.03.3 wheel (`aidensong123/casmi26-offline-rdkit-2026033`)
   installs with internet off (~5 min on 4 cores).
10. **Validation scheme that transfers.** Class-1 sim: mask the query's whole *source
    library* (not just its key — else identical-spectrum retrieval fakes 1.000). Class-2
    sim: mask all spectra of held-out structures (250 NPs + 569 obscure set; analog holds
    0.521→0.516). Never feed pool provenance (train-vs-COCONUT flag scored fake 0.94).
    Shipped test.parquet = enveda-180 slice of train (synthetic, drug-like) — useless for
    validation; hidden set = natural products on timsTOF; use enveda-np-examples as holdout.
11. **megayak "Two Rankers" deltas (worth stealing):** adduct-shifted library match
    (shift reference by precursor Δ; same molecule as [M+Na]+ vs [M+H]+ keeps neutral
    losses; Class-1 0.852→0.872); same-polarity analog restriction; adduct-aware MetFrag;
    ranker rows from 2,250 molecules × 7 libraries (data-starved ranker was bottleneck);
    two-ranker blend because public FP weights saw most libraries (0.49 held-out vs
    0.76–0.82 seen — over-trust leak). DreaMS embeddings as analog channel: no win.
12. **octaviograu "Evidence Ranking" (independent implementation):** sparse fragment +
    neutral-loss index, sim⁴-weighted neighbor fingerprint prototype, tune/report split by
    key hash with bootstrap CIs, hard leakage assertions. Useful as a second code reference.
13. **Dead ends, measured (skip):** NP-likeness prior (worse than random), consensus
    fingerprint over analogs (0.43 vs 0.52 max), per-instrument normalisation, confidence
    gate for answer-not-in-pool (AUC 0.63), 3 refs/structure (0.52→0.49), ±400 Da analog
    window (no gain), explicit formula prediction (only 1.4× reduction), ChEBI/LIPID MAPS
    global add (dilution; denpugovkin's gated-tail variant unproven — tail slots 21–25 only
    if lib-sim < 0.15).
14. **De novo generation is the wrong tool** (inversion tutorial 134 votes — spectra→SMILES
    transformer, BPE-512, 1 epoch demo; avikdas physics-informed transformer similar):
    exact-InChIKey metric punishes generators combinatorially. Only value: extra candidates.

## Compute budgets

4-channel CPU inference ~47–90 min; FPNet training needs GPU (T4, ~12 min–hours; 30 h/week
quota is a known pain point). Pool build ~10 min; RDKit keying ~5 min/4 cores.

## Discussion highlights

Topics confirm: shipped test.parquet is verbatim train rows (bit-identical peak arrays);
train lacks molecule_id/spectrum_id by design; "six variants plateau 0.33–0.34" thread =
ranking saturated, recall is the ceiling (`LB ≈ 0.162·c1 + 0.22·c2`; pool recall ρ is the
lever); open questions on external weights licensing and GPU quota.
