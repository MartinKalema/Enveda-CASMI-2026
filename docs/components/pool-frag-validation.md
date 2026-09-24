# Pool + MetFrag-lite + Validation — fork-034 baseline (LB ~0.33)

Source: `kernels/fork-034/notebook.ipynb` (11 cells, 0–10). Sibling analysis:
`docs/engine-deep-dive.md`, `docs/kaggle-notebooks-survey.md`, `docs/lessons-learned.md`.

---

## 1. How each piece works (with cell references)

### 1A. Candidate pool — cell 5 (`build_candidate_pool`, `CandidatePool`)

- **Composition:** COCONUT 2.0 precomputed shard (`coco_fp.npy/coco_mass.npy/coco_meta.pkl`,
  ~462k InChIKey14) ∪ train structures (~276k unique `inchikey14` from `train.parquet`)
  → **711,705** after `~tr.inchikey14.isin(coco_keys_set)` dedup (cell 5, lines 72–95).
- **Fingerprint:** per structure `ECFP4(4096) ‖ ECFP6(4096) ‖ RDKitFP(2048, maxPath 6) ‖
  MACCS(167)` = 10,407 raw bits → filtered by `fp_bits.npy` to **6,930 informative bits**
  (train frequency in [0.5%, 99.5%]), bit-packed (`np.packbits`) for storage.
  Train half rebuilt at runtime with `MPool(4)` over `fp_and_mass` (~5 min).
- **Index:** `CandidatePool` sorts everything by neutral exact mass; `window(t, ppm)` =
  two `searchsorted` calls (cell 5, lines 40–43). `fps(idx)` = `unpackbits` slice.
- **Query path (cell 9):** molecule target = **median** neutral mass over its 1–16 spectra
  (neutralised via 30-entry adduct table, cell 4 `neutral_mass`); candidates =
  `pool.window(target, PPM_WIN=8.5)`, fallback `±30 ppm` if empty (cell 1, cell 9 lines 29–31).
  Median **~56 candidates**, max ~401. **No cap, no mass-proximity pre-rank.**
- **Why ±8.5 ppm:** tighter beats ±10/±20/±30 (Class-2 MRR 0.524/0.521/0.509/0.500, CFG
  comment cell 1). timsTOF has ~+1.4 ppm systematic offset; 8.5 absorbs it without
  admitting decoys. Fallback 30 ppm fires only on empty windows (adduct-mislabel tail).
- **Why PubChem expansion hurt (0.52 → 0.35, survey §3):** analog-Tanimoto
  (`sim^p·Tanimoto`, cell 8) rewards near-duplicates of analogs; PubChem isomer lists add
  dozens of same-mass near-duplicates per window. Truth is usually a *derivative*
  (median Tanimoto 0.83 to best analog), so decoys outrank it; the FPNet channel
  (`f·z`) degrades worse (0.73 → 0.38) because its logits were trained against COCONUT-scale
  decoy density. Pool is **small on purpose**: recall must come without near-duplicate decoys.
  Same reason `USE_BIO_DB` (ChEBI/LIPID MAPS) ships OFF (cell 1, line 29; denpugovkin's
  gated-tail variant unproven).
- **Cost:** pool build ~10 min; per-query window is O(log N).

### 1B. MetFrag-lite channel — cell 6 (`mol_graph`, `fragment_masses`, `explain_score`, `frag_scores`)

- `mol_graph(smi)`: RDKit mol → per-atom weights (element monoisotopic mass + implicit Hs,
  `AMU` table cell 6 lines 4–8) + bond list. Returns None if any atom outside table.
- `fragment_masses(smi, max_breaks=2, max_bonds=34)`: break every 1 bond, then every pair;
  connected components via `_components` (BFS); fragment mass = weight-sum of each
  component; output = sorted unique masses + intact mass. Molecules with >34 bonds return
  intact mass only (perf guard).
- `explain_score`: ionise each fragment with H-shifts (−2..+2) ± proton by polarity
  (`mode`), match query peaks within `MZ_TOL=0.01 Da` (searchsorted ±1 neighbour),
  score = **sqrt-intensity fraction explained**, max over the molecule's spectra.
- `frag_scores(cands, specs, mode)`: `MPool(4)` over `_frag_masses_wrapper` (chunksize 8),
  ~14 ms/candidate; peaks re-cleaned with entropy weighting OFF.
- **Feature role (cell 8):** 4 frag columns (raw, rank-norm, gap-to-max, z-score) of 31.
- **Numbers:** alone Class-2 MRR 0.259; +analog 0.52 → 0.545; **corr(analog, frag) = 0.058**
  on true scores — near-orthogonal, hence stacks despite weak solo value. Isomer-group MRR:
  frag 0.28 vs analog 0.54 / model 0.49 (engine-deep-dive §2.6).
- **Assumptions/limits:**
  1. Only single/double homolytic-style cleavages; no rearrangements beyond ±2 H
     (no McLafferty, retro-Diels–Alder, water cascades except as H-shift coincidences).
  2. Uniform bond-break cost — no bond-dissociation energies (original MetFrag weights by
     bond energy; lite drops it for speed).
  3. Fragment = atom-subset mass; ignores charge localisation, neutral-loss ordering,
     adduct-specific fragmentation (megayak's adduct-aware MetFrag is the known fix).
  4. Heavy-atom table only (C,H,N,O,P,S,F,Cl,Br,I,Na,K,Si,B,Se); exotic elements → None → 0.
  5. `max_bonds=34` blinds large NPs (exactly the test domain tail).
  6. sqrt-intensity scoring underweights diagnostic low-abundance fragments.
- **Complementarity mechanism:** analog scores *relative-seen* evidence (needs a library
  neighbour within 200 Da); frag scores *this-candidate physics* (needs nothing external).
  Failure modes are disjoint: sparse-library scaffolds keep frag signal; frag-blind
  rearrangement-driven spectra keep analog signal. Corr 0.058 is the empirical proof.

### 1C. Validation + experiment discipline (cells 1, 8; survey §§10–11; lessons-learned)

- **Class-1 (library-masking):** mask the query's whole **source library**, not just its key —
  else an identical spectrum from the same library retrieves itself and fakes MRR 1.000.
- **Class-2 (structure-masking):** mask **all spectra of held-out structures** (250 NPs +
  569 obscure set in top forks); analog holds 0.521 → 0.516 under masking, i.e. transfers.
- **Provenance-leak bans:** never feed pool provenance (train-vs-COCONUT flag scored fake
  0.94); never pre-filter candidates by `lib_sim`-dependent rank (cap-80 killed retention
  0.992 → 0.752 because Class-2 rows have lib_sim=0 by definition).
- **Seed-noise measurement:** GBM fit noise ±0.006 LB / 0.0072 local; protocol = 8-model
  ensemble (2 priors × 4 seeds, cell 8 lines 83–93), distrust deltas <0.007, double-submit
  to confirm. Sample weight W1 ≈ 0.42–0.50 solved from the LB equation
  (LB ≈ 0.162·c1 + 0.22·c2), not the 0.16 raw class share.
- **Single-variable A/B:** one change per submission ("ship what scored"); shipped
  `test.parquet` is a train-slice (synthetic/drug-like) — useless for validation; hidden set
  is timsTOF natural products → validate on `enveda-np-examples` holdout.
- **Diagnostics dashboard (cell 10):** per-molecule `n_candidates / best_library_sim /
  best_analog_sim / top_prob` histograms + lib-vs-analog scatter coloured by posterior —
  reads channel agreement and candidate-multiplicity at a glance.

---

## 2. Assumptions / constraints exploited (all three)

1. **Metric is exact InChIKey14 MRR@25** → retrieval+rerank, never generation (one wrong
   atom = 0; MassSpecGym de-novo top-1 = 0.00 confirms).
2. **Test = MCES-novel timsTOF natural products** → COCONUT (NP chemistry) over PubChem
   (synthetic decoys); instrument-matched analog reps (+0.017).
3. **Adduct labels mostly right, precursor m/z precise to ~ppm** → tight 8.5 ppm window +
   30 ppm fallback instead of wide windows; formula-mass index would tighten further
   (Class-1 0.9206 → 0.9253, dariushafshar).
4. **9 h offline CPU kernel, no GPU, RDKit wheel only** → bitpacked pool, numba kernels,
   1–2-bond frag, precomputed FPNet logits; nothing trainable at inference except GBM
   (pre-fit offline, shipped as `rank_train.npz` + refit in-kernel, cell 8).
5. **Fragmentation is scaffold-local + mass-shiftable** → `max(direct, shifted)` analog
   kernel with `sim^4` suppression of weak shifted matches.
6. **Channels are independent evidence types** (seen / relative-seen / physics /
   learned) → GBM with agreement block learns "trust lib iff model corroborates".

---

## 3. Deep research

### 3a. Pool coverage theory — the recall ceiling

- For retrieval MRR, LB decomposes as `LB ≈ A·c1·ρ1·r1 + B·c2·ρ2·r2` where ρ = pool recall
  (truth in ±window), r = ranker top-25 hit-rate given recall. Top-fork fit:
  `LB ≈ 0.162·c1 + 0.22·c2` (survey §Discussion), so **ρ2 (Class-2 recall) is the lever**:
  every +1 pp of NP recall ≈ +0.0022 LB, vs ranking gains diluted by r2.
- MassSpecGym (Bushuiev et al., NeurIPS 2024) formalises this: retrieval candidates are
  capped at |C| ≤ 256 same-mass structures; de-novo baselines score ~0 while
  database-constrained methods scale with DB coverage. Novel-NP recall at ±10 ppm from a
  711k NP-biased pool is empirically ~0.99 *of known chemistry* but structurally novel
  test NPs are by definition underrepresented — the ceiling is chemical, not code.
- Math: if hidden set has fraction ν of truly-novel scaffolds absent from COCONUT∪train,
  ρ2 ≤ 1−ν regardless of window. Widening the window cannot recover ν (adds decoys only:
  0.521 → 0.500 at ±30 ppm); only pool *composition* (more NP DBs, gated to tail slots)
  or generative fill (slots 16–25) raises ρ2. Hence the ranked upgrades below put
  recall levers (formula-mass index, gated tail, generative fill) alongside rank levers.

### 3b. In-silico fragmentation SOTA beyond 1–2-bond cleavage

| Method | Idea | Fits 9 h CPU kernel? |
|---|---|---|
| **CFM-ID 4.0** (Wishart; competitive fragmentation modelling) | Learns bond-breakage transition probabilities (EM over spectra), predicts spectra per candidate; strong at low CE, weak at high CE | No — per-candidate prediction too slow in-kernel; use offline to pre-score or distil |
| **ICEBERG** (Goldman et al. 2024) | GNN predicts fragmentation DAG (which bonds break) + fragment intensities; SOTA on MassSpecGym spectrum-simulation | No GPU at inference; but **precompute offline** fragment sets per pool structure is viable |
| **FraGNNet** (probabilistic fragment graph, NIST20 eval) | Combinatorial fragment enumeration + learned intensity head with uncertainty | Same verdict as ICEBERG: offline precompute, online lookup |
| **MetFrag full** (bond dissociation energies + bond-order penalties) | Lite + energy-weighted scoring, deeper (3-bond) trees | Yes — drop-in: add BDE weights + depth-3 for small candidates only |
| **ChemFrag / MAGMa / MS-FINDER** | Rule-based rearrangements, substructure-restricted cleavage | MAGMa-style H-rearrangement rules are the cheap portable win |

- **Verdict:** nothing learned fits the CPU kernel at inference; the practical SOTA path is
  (i) cheap rule upgrades in-kernel (BDE weights, adduct-aware ionisation, neutral-loss
  ladder, depth-3 for <25-bond candidates), plus (ii) offline-distilled scores (ICEBERG/
  FraGNNet fragment bags precomputed per pool structure, shipped as `.npy`, looked up in
  cell 9). The 0.058-corr complementarity argues for keeping frag as an independent
  physics channel, not folding it into the neural channel.

### 3c. Validation methodology literature

- **Leakage taxonomy (MassSpecGym §splits; Bushuiev):** 2D-InChIKey splits leak via
  Tanimoto > 0.85 near-dupes across folds → demand MCES ≥ 10 separation. Our fork's
  library-masking (Class-1) + structure-masking (Class-2) is the same idea at evaluation
  time; minimum bar is InChIKey14-disjoint, preferably scaffold/MCES-disjoint.
- **When local CV transfers:** transfers iff (a) holdout matches hidden chemistry
  (NP, not synthetic train-slice), (b) instrument matches (timsTOF reps), (c) no
  provenance features, (d) n ≥ 100 × ≥3 seeds (lessons 21–23: n=30 single-seed swung
  50×). The top forks' "250 NPs + 569 obscure" holdout + seed-bagged GBM + double-submit
  protocol is exactly this checklist; our v4's 0.013 LB burn is the documented
  counterexample (uncalibrated split, shipped anyway).
- **Reporting:** tune/report split by key hash with bootstrap CIs (octaviograu fork);
  hard leakage assertions in code (assert no test key in library index).

---

## 4. Concrete upgrades, ranked by expected LB delta

1. **Adduct-shifted library match** (megayak; shift reference by precursor Δ so
   [M+Na]+↔[M+H]+ share neutral losses): Class-1 0.852 → 0.872 ≈ **+0.003 LB**. Cheap.
2. **Formula-mass library index** (`LIB_MASS_FORMULA`, 30-entry adduct table already in
   cell 4): Class-1 0.9206 → 0.9253 ≈ **+0.001–0.002 LB**, shrinks windows (fewer decoys).
3. **Adduct-aware + BDE-weighted MetFrag** (energy weights, same-polarity analog
   restriction, depth-3 for small cands): frag 0.259 → ~0.30 est. ≈ **+0.002–0.004 LB**,
   keeps corr low. Medium effort.
4. **Gated tail slots 21–25** (denpugovkin: ChEBI/LIPID MAPS or generative fill only if
   lib-sim < 0.15): raises ρ2 without PubChem-style dilution ≈ **+0.002–0.005 LB** if ν > 0.
5. **Instrument-matched analog reps + per-molecule fusion audit** (median mass, max sims,
   mean-of-views logits): +0.017 analog / +0.019 fusion locally ≈ **+0.002 LB** combined.
6. **Two-ranker blend** (public-FP-weights ranker + held-out ranker; megayak 0.49 vs
   0.76–0.82 seen/held-out gap): fixes FPNet over-trust ≈ **+0.002 LB**.
7. **Offline-distilled ICEBERG/FraGNNet fragment bags** (precompute per pool structure,
   ship `.npy`, lookup in cell 9): bigger frag lift ≈ **+0.003–0.006 LB**, one GPU-day
   offline, zero inference cost. Best effort-adjusted bet after 1–3.
8. **Do NOT do:** PubChem expansion (−0.17), candidate caps by mass (−25% Class-2 recall),
   fixed lib+analog weights (0.52 → 0.27 Class-2), DreaMS analog embeddings (no win),
   NP-likeness priors, ±400 Da windows, confidence gating (AUC 0.63).

*Do NOT commit — analysis only.*
