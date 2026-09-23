# Engine deep-dive: the top CASMI26 retrieval engine (prvsiyan family)

Source code on disk: `/tmp/topengine/` (pulled 2026-09-23 via `kaggle kernels pull`).
Primary document: `prvsiyan/analog-propagation-casmi-2026-baseline.ipynb` (37 cells, 17 code, ~1122 code lines).
Forks compared: `beraterolelk/…analog-ranker.ipynb` (0.336), `haideptry-4ch/…4-channel…ipynb` (0.339),
`haideptry-cosine/…` (same code, 0.339), `malik/…quad-channel…`, `flexonafft/…strong-retrieval…`,
`megayak/…two-rankers…`. All forks are the same engine with constant/delta changes; prvsiyan's
notebook is the living lab report and the only one that documents its own failed experiments.

Public LB: prvsiyan 0.335, forks 0.336–0.339. Ours: ~0.115. Gap ≈ 0.22, nearly all of it ranking
Class-2 molecules (structures with no reference spectrum) that we currently cannot order.

## 1. Architecture in one page

Retrieve from a fixed pool, then spend everything on ordering. No generation anywhere.

- **Pool (cells 9–11):** train structures (275,810 InChIKey14) ∪ COCONUT 2.0 (462,028) → **711,705**,
  deduped on InChIKey14 (metric is stereo/tautomer-blind), sorted by exact mass. Fingerprint per
  structure `ECFP4(4096) ‖ ECFP6(4096) ‖ RDKitFP(2048) ‖ MACCS(167)`, filtered to bits with
  train frequency in [0.5%, 99.5%] → **6,930 bits**, bit-packed. COCONUT half ships precomputed
  (`coco_fp.npy` etc.); train half rebuilt at runtime (~5 min, 4 workers). ChEBI+LIPID MAPS
  (`USE_BIO_DB`) ships OFF in prvsiyan, ON in beraterolelk.
- **Query mass (cell 7):** 30-entry adduct table (incl. dimers/trimers, water losses, MeOH/MeCN
  adducts); neutral mass per spectrum = `(mz*z − d)/n`; molecule target = **median** over its
  1–16 spectra. Library indexed on **formula mass** (exact arithmetic) with precursor fallback
  (`LIB_MASS_FORMULA=True`), which beats `max(10ppm, 0.01Da)` (Class-1 0.9206→0.9253, narrower).
- **Cleaning + similarity (cell 6):** floor 0.002 of base peak, top-256, intensity power **1.0**
  (beats sqrt 0.919 vs 0.895), **entropy weighting** (Li et al. 2021: if spectral entropy S<3,
  reweight `w=0.25+0.25S`), 0.01 Da match tolerance. All in numba (`_clean`, `entropy_sim`,
  `entropy_sim_shift`, `search`/`search_shift` with `prange`).
- **Ch1 library search:** ±10 ppm neutral window on the library index; entropy sim; max over the
  molecule's spectra → per-structure score. Alone: Class-1 ≈0.93, LB 0.151 (Class-1 value
  A≈0.162 follows).
- **Ch2 analog propagation (the main idea, cells 5+12, `analog_sim`):** for each query, take
  library representatives within **±200 Da** of target, shift each reference spectrum by
  `target − rep_mass`, score `max(direct, shifted)` entropy sim, keep top-**200**. Candidate score
  `max_a sim(a)^p · Tanimoto(f_c, f_a)`, **p=3**. One representative per structure, **preferring
  the test's own instrument** (0.5241→0.5412). Alone: Class-2 0.52–0.55 (vs 0.164 by mass error).
- **Ch3 MetFrag-lite (cells 14–15):** break every 1–2 bonds (≤34 bonds), connected components,
  ±2 H rearrangements, score sqrt-intensity fraction explained within 0.01 Da, max over spectra.
  ~14 ms/candidate. Alone 0.259; correlation with analog only 0.058 → stacks (+analog → 0.545).
- **Ch4 spectrum→fingerprint transformer (cells 1, 17–18, 29):** `FPNet` (d=512, 6 layers, 8
  heads): log-spaced sinusoidal m/z + neutral-loss embeddings, 128 window-diversified peaks
  (`prep_peaks`), precursor/adduct/instrument/CE/mode global token, CLS+mean head → 6,930 logits
  `z`. Rank by **`f·z`** (exact Bayes LL up to a candidate-independent constant — no sigmoid).
  Trained with BCE + **63 same-±10ppm-window decoys under softmax CE**, best-by-validation
  checkpoint, augmentation (peak dropout, intensity jitter, ±5 ppm). Two views averaged:
  per-spectrum model + merged-peaklist model (each fed only its own training distribution).
  Alone 0.468–0.49; +analog → 0.567; all four → 0.612 (local).
- **Ranker (cells 12, 26):** 31 features (5 library incl. rank/max/gap/hit-flag, 9 analog incl.
  max/mean/top-tan/p¹-vs-p³, log n_cand, 6 model raw+z/rank, 4 frag, **6 cross-agreement**:
  lib·(1−model-rank), analog·(1−model-rank), corroboration flags, lib–model correlation).
  `HistGradientBoostingClassifier` (depth 6, 500 iter, lr 0.03, leaf 80, l2 1.0), **8 models
  (2 priors × 4 seeds) averaged**, sample weight W1=0.50 on Class-1 rows. Seed noise ±0.006 —
  bagging is load-bearing.
- **Inference (cell 27):** per molecule: target → lib_hits → analogs (top-200) → pool window
  (±10 ppm, fallback ±30) → optional cap-500 guard ranked by evidence (never mass) → fingerprints
  → frag scores → 31 features → mean GBM proba → top-25 SMILES. Fallback `CCO`. Runtime ~47–90 min
  CPU; FPNet training needs GPU (~2 h, T4); pool build ~10 min.
- **Budgets:** median 52–56 candidates/molecule, max ~401 → cap 500 never fires; full test ≈
  1213 spectra / ~400–1200 molecules in one run.

## 2. Why each piece works (first principles)

1. **Retrieval, not generation, because the metric is exact-structure MRR@25.** A generator must
   emit the exact tautomer-canonical skeleton character-perfect; one wrong atom = 0 for that slot.
   A retriever with a 711k pool containing the answer needs only to *order* ~56 candidates.
   MassSpecGym's 0.00 top-1 for all de-novo baselines (our docs/constraints-and-literature.md §"numbers")
   is confirmed: the engine's inversion-tutorial fork is a demo, never the scorer.
2. **Analog propagation works because fragmentation is scaffold-local + mass-shiftable.**
   Congeners share a core; substituent changes shift a subset of fragment masses by a near-constant
   ΔM while core fragments stay put. `max(direct, shifted)` entropy sim therefore matches relatives
   the library never saw; `sim^p·Tanimoto` then propagates the relative's *known structure* onto
   same-mass candidates, with p=3–4 suppressing weak shifted matches that otherwise flood the max.
   Self-match at ΔM=0 covers Class-1 for free — no Class-1 branch needed.
3. **`f·z` is not a trick — it is the Bayes readout, which is why hard in-window decoys train it.**
   `Σ f·logσ(z)+(1−f)·logσ(−z) = f·z + const` exactly, so the dot product is the full Bayesian
   score and one matmul at inference. Softmax CE against 63 *same-window* decoys trains the actual
   ranking task (distinguishing same-mass isomers), not bit reconstruction; random-decoy or BCE-only
   training teaches an easier, non-transferring task. Memorisation (~276k structures) forces
   best-by-validation checkpoints + augmentation.
4. **Tight window + no cap works because mass is a filter, not evidence.** Inside ±10 ppm
   (0.0085 Da at 500 Da) mass proximity carries ~zero information, and Class-2 answers have
   lib_sim=0 *by definition* — any `lib_sim·100 − |Δmass|` pre-rank deletes them quasi-randomly
   (retention 0.992→0.752 at cap 80). Same reason ±10 beats ±20/±30: wider windows add only decoys.
   Formula-mass indexing tightens further by deleting precursor/adduct error modes (riken 0.005 Da
   precision, gnps Na↔H mislabels) rather than widening to absorb them.
5. **GBM calibration works because the channels are incommensurable and class-asymmetric.**
   Fixed lib+analog weights collapse Class-2 (0.52→0.27): every Class-1 hit is a wrong isomer for a
   Class-2 query. The GBM learns "trust lib when model corroborates, else trust analog/model/frag"
   via the agreement block; W1≈0.42–0.50 solves the leaderboard equation
   (LB≈0.162·c1+0.22·c2), *not* the raw class share 0.16, because Class-2 value is discounted by
   pool recall. Seed-bagging removes ±0.006 fit noise that otherwise masquerades as signal.
6. **The four channels stack because they are independent evidence types** (compare / borrow /
   derive / predict): library (same molecule seen), analog (relative seen), frag (physics of this
   candidate), model (learned spectrum→substructure). True-score correlation analog–frag is 0.058;
   each pair adds measured MRR. Isomer-group MRR (unconfounded, cell 33): analog 0.54, model 0.49,
   frag 0.28, library −0.05 (anti-informative on Class-2 — correctly ignored by the ranker there).

## 3. Map to our research docs

- **docs/constraints-and-literature.md — mostly CONFIRMED, two corrections.**
  Confirm: retrieval+rerank over generation (engine §§ cells 0/31–32 + our §2.1); hard same-mass
  decoys + softmax (MIST-CF Eq 4 → engine cell 29 verbatim); instrument/adduct conditioning
  (FPNet tokens, instrument-matched reps); slot-dedup hygiene (metric-exact Key14, megayak).
  Correct: (a) "formula-first funnel" is NOT what the engine does — candidates come from the mass
  window directly and formula never appears as a feature; mass-error-as-feature was tried and is a
  wash (cell 36). (b) "subformula-labelled peaks (v1)" direction is validated only as MetFrag-lite
  bond-breaking (0.259, stacks); our v1 whole-spectrum subformula ranker failing does not
  contradict MIST — MIST uses subformulae as *encoder inputs*, we used them as the *scorer*.
- **docs/casmi-history-survey.md — CONFIRMED with one reordering.** Fingerprint retrieval as
  primary + fragmentation as rerank feature (finding 2) is exactly Ch4 + Ch3. Library-anchors-first
  (finding 5) is Ch1 + agreement features. Diverse-scorer consensus (finding 3) is the GBM over 4
  independent channels — but history's "exactly two diverse scorers" understates it: four stack
  monotonically here (0.52→0.545→0.567→0.612). Database boosting (finding 4) is *inverted*: PubChem
  expansion catastrophically dilutes (0.52→0.35, cell 8/34) — the engine's pool is small on purpose.
- **docs/limits-analysis.md — CONFIRMED, numbers sharpened.** MRR rank math, mass-funnel
  reliability (median spread ~0 ppm), pool sizes (median ~56 at ±10 ppm vs our measured 78–188 at
  wider windows), formula-bottleneck shape (median 46 same-formula isomers, cell 33), timsTOF
  domain gap (upweighted/matched explicitly), rare-adduct thinness (adduct-aware MetFrag +
  adduct-shifted lib in megayak), peak-density shift (top-256 + 128 window-diversified truncation).
  Bound update: ceiling A+B≈0.43 with ρ≈1 → remaining ≈0.10 is *ranking*, not recall ("recall is not
  the bottleneck", cell 34) — contradicts our lesson 12 ("ranker < recall").
- **docs/lessons-learned.md — CONFIRMED and extended.** Anchor-gated multi-seed validation
  (rules 21–27) is the engine's grouped-by-query CV + "ships what scored" policy (cells 23–25);
  never-replace-the-floor (rule 26) matches the cap post-mortem; seed-noise band (rule: distrust
  <0.007) is engine cells 24–25 (0.0072 local, 0.006 LB, double-submit protocol). Extension: our
  rule 28 "entropy is DEAD" is **contradicted** — entropy+power-1.0 beats cosine 0.919 vs 0.895
  *under their cleaning*; the transfer failure was our cleaning/intensity power, not the kernel.
  Our rule "ranker < recall" is contradicted (see above); "drop entropy, stop retrying" should be
  rescoped to "match their cleaning before judging".

## 4. What THEY thought of that WE did not (ranked by likely LB value)

1. **Analog propagation `max sim^p·Tanimoto`, |ΔM|≤200, top-80–200, p=3–4** (cells 5, 12) — our v4
   tried it but replaced the cosine floor on an uncalibrated split and shipped LB 0.013; engine
   keeps it as a *feature block inside a calibrated ranker*. Expected: the single biggest lever
   (+0.3 Class-2 MRR locally; 0.16→0.52).
2. **GBM ranker over ~25–31 cross-channel features incl. agreement block, seed×prior bagged,
   W1 solved from LB** (cells 12, 22, 26) — our v6/v9 rankers lack the analog-max/mean/top-tan
   features, the lib–model corroboration features, seed bagging, and LB-solved W1. Expected: large;
   hand-weighting lib features demonstrably destroys Class-2.
3. **Spectrum→fingerprint transformer trained with 63 in-window hard decoys, ranked by raw-logit
   `f·z`, best-val checkpoint, dual input views** (cells 1, 17–19, 29) — our v2/v8 attempts had no
   hard negatives, wrong/weak checkpointing, single view. Expected: large (0.468 alone).
4. **Tight ±10 ppm window with NO candidate cap** (cells 2, 21) — our windows are ±20 ppm and we
   cap by mass; engine measures cap-80 deleting 25% of Class-2 answers. Expected: medium-large
   (removing an 80-cap ≈ +0.055 predicted LB).
5. **Entropy similarity (power 1.0 + S<3 weighting) as the spectral kernel** (cell 6) — we
   declared entropy dead (lesson 28) without their cleaning. Expected: medium (Class-1 0.893→0.93).
6. **MetFrag-lite 1–2-bond cleavage ±2H, sqrt-intensity explained fraction** (cells 14–15) — our
   v1/v-alt subformula enumeration is the weaker variant; bond-breaking stacks +0.024 with analog
   at corr 0.058. Expected: medium-small, independent.
7. **Instrument-matched analog representatives + formula-mass library index** (cells 7, 20) —
   our v7 planned these; engine measures +0.017 analog and +0.005 Class-1. Expected: small-medium,
   mechanical.
8. **Per-molecule fusion: median neutral mass, max lib/analog sims, mean-of-views logits**
   (cells 18, 27) — we fuse by max/default; engine's view-specific averaging is measured +0.019 LB.
   Expected: small.
9. **COCONUT-only pool discipline (no PubChem), InChIKey14 dedup, 6930-bit informative fingerprint,
   precomputed + mass-sorted index** (cells 9–11) — we have COCONUT but without the PubChem
   autopsy, bit filtering, or sorted-window structure. Expected: small positive + large avoided loss.
10. **Validation machinery: grouped-by-query CV, leakage bans (provenance features, source-library
    masking), noise-floor double-submit, one-change-per-submission, ship-what-scored** (cells
    22–25) — process, not LB directly, but it is why their constants transfer and ours did not.
11. **megayak deltas: adduct-shifted library match, same-polarity analogs, adduct-aware MetFrag,
    2250-mol×7-library ranker rows, metric-exact dedup, two-ranker blend** (megayak cells 0–10) —
    each small (+0.01–0.02 Class-1, data-starved-ranker fix); bundle is the cheapest honest gain.
12. **Negative results worth not retrying:** DreaMS embeddings for analogs (no win, cell 31),
    cross-encoder over (z,f) without peaks (4 failures, cell 32), consensus fingerprints, CE/mode
    matching, 3-reps/structure, ±400 Da, NP-likeness, confidence gating, formula prediction —
    each documented with n and mechanism, saving us GPU-days.

## 5. Runtime / data-structure cheat sheet (for the implementer)

- Library arrays: offset/mz/it (float32), neutral-mass sorted index (`order`, `snm`, `n_ok`);
  per-query `lib_window`/`search` are numba `prange` loops — no Python per-spectrum overhead.
- Pool: `pool_mass.npy` (sorted float), `pool_fp.npy` (bit-packed uint8), `pool_meta.pkl`
  (keys/smiles/nbits); `window()` = two `searchsorted` calls; `fps()` = `unpackbits` slice.
- Analog reps: `build_rep` → `(rep, rep_key, rep_nm)` mass-sorted; `analog_sim` shifts by
  `target − rep_nm` and calls `search_shift` (best of direct/shifted per pair).
- Frag: `fragment_masses` per candidate (multiprocessed), `explain_score` per spectrum, max.
- Model: `prep_peaks` → batched FPNet → per-spectrum mean + merged-list view → mean.
- Ranker: `rank_features` (31 cols) → `rank_proba` (mean of 8 GBMs) → argsort top-25.
- Files: train-side `fp_bits.npy`, `coco_*`, `rank_train.npz`, `fp_*.pt` (single+merged),
  offline RDKit wheel; output `submission.csv` (25 `;`-joined SMILES).
