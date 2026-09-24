# Component: Library Search (Channel 1 — direct entropy match)

Baseline: `kernels/fork-034/notebook.ipynb` (fork of LB 0.328 leader, V17 4-channel engine).
This doc covers ONLY the direct library-search channel (`lib_sim`). Analog propagation,
MetFrag-lite, FPNet, ranker are separate components.

## 1. How it works, line by line

### 1a. Hyperparameters (cell 1, `CFG`)

| Param | Value | Measured effect (their comments) |
|---|---|---|
| `PPM_WIN` | 8.5 ppm | ±10 ppm → 0.521, ±8.5 ppm → 0.524 Class-2 MRR; tighter = fewer decoys |
| `PPM_FALLBACK` | 30 ppm | only if tight window is empty |
| `INT_FLOOR` | 0.002 | drop peaks < 0.2% of base peak |
| `MAX_PEAKS` | 256 | keep 256 most intense survivors |
| `MZ_TOL` | 0.01 Da | peak-alignment tolerance |
| `INT_POWER` | 1.0 | linear intensity beats sqrt (0.919 vs 0.895 — their ablation) |
| `ENT_WEIGHT` | True | Li et al. 2021 low-entropy reweight |

### 1b. Spectral kernels (cell 3, numba `njit`)

`_clean(mz, it, floor, topk, power, ent_weight)` — identical cleaning applied to query
AND every library candidate (query once in `lib_sim`, candidates inside `search`):
1. threshold at `floor * base_peak`;
2. if survivors > `topk`, keep `topk` most intense (restored to m/z order);
3. intensity transform `it^power`, L1-normalize to a probability distribution;
4. if `ent_weight` and spectral entropy `S = -Σ p·ln p < 3.0`: raise to power
   `w = 0.25 + 0.25·S`, renormalize. Low-entropy spectra (few dominant peaks) get
   flattened so a single base-peak coincidence cannot dominate — this IS the Li 2021
   "weighted entropy" prescription, with S<3 cutoff from the paper's S=1–3 density band.

`entropy_sim(qmz, qp, cmz, cp, tol)` — unweighted Li 2021 similarity:
1. `SA`, `SB` = entropies of the two full cleaned distributions;
2. two-pointer m/z walk (`tol`): matched pairs contribute `qp+cp` as ONE merged peak,
   unmatched peaks contribute their own intensity as singletons into `buf`;
3. `SAB` = entropy of `buf/tot` (the mixture distribution over ALL peaks);
4. return `1 - (2·SAB - SA - SB)/ln(4)`, i.e. 1 minus normalized Jensen–Shannon-type
   divergence. Range ≈ 0..1.

`entropy_sim_shift(...)` — `max(direct, shifted)`; shift≈0 short-circuits to direct,
so Channel 1 pays zero extra cost (Channel 2 passes nonzero shifts).

`search(...)` — `prange` parallel loop over candidate spectrum indices: slice
`allmz/allin` via `off`, `_clean`, `entropy_sim`. `search_shift` is the shifted twin.

### 1c. Library construction and index (cell 4)

- `MASS`/`ADDUCTS`: 30-entry adduct table (incl. dimers/trimers, water losses,
  MeOH/MeCN adducts, doubly charged). Electron mass handled explicitly
  (`PROTON = H - E`).
- `neutral_mass(mz, adduct)`: per-spectrum neutral mass `(mz·z − d)/n`, NaN if adduct unknown.
- `load_library(train.parquet)`: flat `off/mz/it` float32 arrays + `prec`, `ad`, `ik`
  (inchikey14), `best` smiles map; neutral masses sorted once (`order`, `snm`, `n_ok`;
  NaN pushed to `1e18` tail).
- `lib_window(L, target, tol)`: two `searchsorted` calls on the sorted neutral-mass
  index — O(log N) candidate prefilter, no Python per-spectrum loop.
- `build_rep`: one richest (most peaks) spectrum per structure for analog retrieval
  (Channel 2, not used by `lib_sim` itself).

### 1d. The Channel-1 scorer (cell 6, `lib_sim(L, specs, target)`)

```
cand_all = lib_window(L, target, target·PPM_WIN/1e6)   # ±8.5 ppm neutral window
for each query spectrum (mz, it, adduct):
    cand = cand_all[same adduct]  (fallback: all, if empty)
    qm, qp = clean_spectrum(...)  (= _clean with CFG params)
    sc = search(qm, qp, cand, ...)  # entropy_sim per candidate
    agg[structure] = max over spectra AND over replicate library spectra
return agg  # {inchikey14: best entropy sim}
```

Three design choices do heavy lifting: (i) **adduct gating** (same-adduct refs first —
avoids H/Na mislabel cross-matches); (ii) **max over the molecule's 1–16 spectra**
(multi-energy/instrument consensus for free); (iii) **max over replicate library
spectra per structure** (instrument/CE diversity absorbed, not averaged away).
Called in cell 9 per test molecule with `target = median(neutral masses)`; output
`lv` vector becomes 5 of the 31 ranker features (value, rank, max, gap, hit-flag).

## 2. Assumptions and where each breaks

1. **Answer's neutral mass is within ±8.5 ppm of truth.** Breaks on: wrong adduct
   call (Na↔H swaps ≈ 21.98 Da, not ppm-scale — mitigated by 30-adduct table, not
   eliminated); dimer/trimer misassignment; precursor m/z rounding on low-res
   instruments. Fallback ±30 ppm catches only the mild cases.
2. **Same structure ⇒ similar MS/MS (fragmentation is deterministic given
   molecule+adduct+energy).** Breaks across collision energies (their CE spread is
   real; max-over-spectra mitigates), across instrument families (timsTOF vs
   Orbitrap fragmentation differs — adduct gate + replicate-max only partly cover;
   Channel 2 adds instrument-matched reps for this reason), and for fragile/adduct-
   specific fragmentation (water losses, in-source decay).
3. **Entropy similarity is the right divergence.** Assumes cleaned spectra are
   faithful probability distributions over fragments. Breaks when: spectra are
   noise-dominated (test median 230 peaks vs train 42 — the 0.002 floor + top-256
   cap exist precisely for this); S<3 weighting assumes low-entropy = few real
   fragments, but a spectrum with 3 noise spikes also has low S and gets flattened
   into false confidence.
4. **Library contains the answer (Class-1 regime).** By definition false for
   Class-2 (novel) molecules — `lib_sim` returns near-0 there and the ranker must
   learn to ignore it (their agreement features exist because fixed lib weights
   collapse Class-2 0.52→0.27). Any prefilter/truncation on `lib_sim` deletes
   Class-2 answers quasi-randomly (their cap-80 autopsy: retention 0.992→0.752).
5. **0.01 Da alignment is sufficient.** Assumes calibrated high-res data; breaks on
   IT/low-res spectra or systematic offsets (their +1.4 ppm timsTOF offset note —
   window is tight enough that calibration bias matters).

## 3. Competition constraints it exploits

- **Metric is exact-InChIKey14 MRR@25, retrieval-scored.** No generation: the pool
  (711k) contains most answers, so a same-mass identity hit at rank 1 banks ≈1.0.
- **Test = mixed knowns + novels; train spectra ARE the library.** 2.54M reference
  spectra including near-duplicates of Class-1 queries → identity search saturates
  (Class-1 ≈ 0.87–0.93 alone). Our local "pooled-truth retrieval 0.97" measured
  the same saturation.
- **Median ~52 candidates at ±8.5 ppm.** The window is a filter, not evidence:
  tight window deletes decoys at zero recall cost (median count, max ~401).
- **9h CPU kernel, no GPU needed for this channel.** Entire channel is numba
  `prange` + `searchsorted` — runs in minutes; GPU budget reserved for FPNet training.
- **Multi-spectrum molecules (1–16 spectra).** Max-aggregation turns CE/instrument
  spread from a liability into coverage.
- **Adduct diversity with thin rare-adduct support.** 30-adduct physics table +
  same-adduct gating extracts signal where learned models starve.

## 4. Literature: is there a strictly better similarity?

**Li et al. 2021, Nat Methods (entropy similarity, n=434k NIST20 spectra, 43 algorithms).**
Result: entropy similarity beat all 42 alternatives incl. dot product; FDR <10% at
0.75 cutoff where dot product managed ~15% at >0.95. Prescription: L1-normalize,
compute mixture entropy over ALL peaks (matched merged + unmatched singletons),
apply `w=0.25+0.25S` weighting when S<3, use 0.01–0.02 Da tolerance. Fork-034
implements this verbatim (cell 3). Follow-ups (Li/MatchMS `FragmentScores`,
GNPS integration) confirm: entropy dominates for **identity/library-matching**;
the S<3 weighting and noise-floor cleaning are the load-bearing details, not the
divergence formula itself.

**Spec2Vec (Huber 2021, unsupervised Word2Vec on peak co-occurrence).**
High Spec2Vec ⇒ higher structural (Tanimoto) similarity than high cosine among top
0.1% pairs; much faster than cosine at scale. But: it predicts *structural*
relatedness, not identity — strictly the wrong objective for Channel 1, and it
needs a large trained embedding + vocabulary. Their engine tried embeddings for
analogs (DreaMS) with no win (engine-deep-dive §12).

**MS2DeepScore (Huber 2021, supervised Siamese net predicting Tanimoto from spectra).**
Beats Spec2Vec and modified cosine at finding structurally related pairs
(precision/recall on 3601-spectra test). Again an **analog-task** metric: superior
when the answer is NOT in the library. For identity search it has never beaten
entropy in benchmarks; the 2026 Turkina et al. 20-metric benchmark clusters
MS2DeepScore separately from entropy/cosine family and confirms task-dependence:
entropy-family for identity, learned scores for analog retrieval.

**Verdict: no strictly better identity similarity.** For Channel 1 (same-mass,
same-adduct, answer-in-library), properly-cleaned entropy similarity remains the
SOTA kernel; their 0.919 vs 0.895 ablation over sqrt-cosine reproduces the paper.
Upgrades live in cleaning/tolerance/aggregation, not in replacing the kernel.
The genuinely better tool for Channel 2 (answer NOT in library) is a different
question — mass-shifted entropy is a heuristic there, and MS2DeepScore/Spec2Vec
as *analog pre-rankers* remain untested in this engine.

## 5. Upgrade proposals (offline, 9h CPU), ranked by expected LB delta

1. **Port Channel 1 verbatim (+0.03–0.05 LB est.).** Copy cell 3 kernels + cell 4
   adduct table/index + `lib_sim` aggregation exactly (floor 0.002, top-256,
   power 1.0, ent-weight, tol 0.01, adduct gate, max-over-spectra). Our current
   scorer differs in every constant (see §6) — fix by replacement, not tuning.
2. **Tighten window to ±8.5–10 ppm, remove any candidate cap (+0.01–0.05).**
   Mechanical; their cap-80 autopsy shows caps delete Class-2 answers while adding
   nothing to Class-1.
3. **Adduct-table expansion to 30 entries (+0.005–0.015).** Dimers/trimers/water
   losses/MeOH-MeCN currently unhandled in our `neutral_mass`; each fixed adduct
   converts an unanswerable molecule into a ±8.5 ppm hit.
4. **Per-spectrum-hit features instead of single score (+0.005–0.01).**
   Feed rank/gap/hit-count/n-spectra-agreeing into the ranker (their 5 lib features)
   rather than one max value — lets the ranker separate confident identity hits
   from lone-peak coincidences.
5. **Tolerance/cleaning micro-sweep on a calibrated split (+0.002–0.008).**
   Grid: floor {0.001,0.002,0.005}, topk {128,256,512}, tol {0.005,0.01,0.02},
   power {1.0} fixed, S-cut {2.5,3.0,3.5}. Cheap (numba), one-change-at-a-time.
6. **Instrument-gated library scoring (+0.002–0.005).** Prefer same-family refs
   (their Channel-2 instrument match gave +0.017 analog; expect smaller for
   identity where replicates already cover it).
7. **Do NOT: replace kernel with Spec2Vec/MS2DeepScore for Channel 1 (−, avoids loss).**
   Learned scores need GPU training, large vocabularies, and optimize the wrong
   (analog) objective here. Revisit only as Channel-2 pre-ranker with CPU
   embeddings precomputed offline.

## 6. Why OUR entropy variant died (0.527 vs cosine 0.970)

Reference: `v4/channels.py:43-52` vs fork-034 cell 3; validation `v7/validate_floor.py:63`.

The exact, fatal difference — **we drop all unmatched peaks** (`v4/channels.py:32-52`):
ours greedy-matches peaks, then computes `1-(2H(A+B)-H(A)-H(B))/ln4` over the
**matched subset only**, with a `<3 matches → 0.0` gate (`v4/channels.py:48-49`).
Li/theirs compute the mixture entropy over the **full merged peak list**: matched
pairs summed, every unmatched peak kept as a singleton (`buf`, cell 3
`entropy_sim`). Unmatched peaks are the mismatch penalty — without them, any two
spectra sharing 3 coincidental peaks score ~1.0 regardless of the other 200 peaks,
and spectra sharing 2 strong peaks score 0.0 by fiat. The test is then
discrimination-free noise, which is exactly what 0.527 measures.

Compounding differences, each verified in source:
- Cleaning `v4/channels.py:10-19`: floor 1% (vs 0.2%), top-500 unsorted-truncated
  (vs 256), no L1 normalization to a distribution, no intensity power handling, no
  S<3 entropy weighting. On peak-dense test spectra (median 230 vs train 42)
  the 1% floor keeps an order of magnitude more noise than theirs.
- Tolerance 0.02 Da (ours) vs 0.01 Da (theirs) — doubles random-match rate in
  dense spectra.
- Scoring context: our validation compared single spectrum pairs globally
  (`v7/validate_floor.py:40-47`); theirs gates by ±8.5 ppm neutral window +
  same-adduct + max over 1–16 query spectra and replicate refs. Cosine
  (`v2/blend.py:20-33`) survived our validation precisely because it normalizes by
  full-spectrum norms (unmatched peaks DO penalize via the denominator) — the one
  property our entropy port discarded.

Fix = port, not tune: lesson 28 ("entropy is DEAD") is contradicted once cleaning
is matched (engine-deep-dive §2). Rescope to "match their cleaning before judging".
