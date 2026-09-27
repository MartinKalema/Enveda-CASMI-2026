# MetFrag-lite fragmentation gaps — fork-034 audit (no commit)

Source: `kernels/fork-034/notebook.ipynb` cell 6 (`mol_graph`, `_components`,
`fragment_masses`, `explain_score`, `frag_scores`); replication `v7/metfrag.py` +
`v7/validate_metfrag.py`; limits doc `docs/components/pool-frag-validation.md`.
Measured channel value: solo Class-2 MRR 0.259; +analog 0.52 → 0.545;
corr(analog, frag) = 0.058 (orthogonal, keep independent).
Frequency estimates below: NP-band = COCONUT 180–470 Da sample (n=2000, matches
hidden timsTOF NP mass range); train-proxy = train structs in test neutral-mass
band (n=2000, drug-like, for contrast). Test: 400 mols, m/z 245–460, all timsTOF,
CE {20,40,60}, adducts 79% [M+H]+ / 16% [M-H]- / 2.6% formate / 1.8% [M+Na]+ /
0.5% other.

## Gap table

| # | Gap in fork-034 cell 6 | Chemistry omitted | Frequency (NP-band / train-proxy) | Concrete fix | CPU in 9 h kernel | Rank |
|---|---|---|---|---|---|---|
| G1 | No BDE ordering — every 1-/2-bond break equiprobable, `explain_score` unweighted count | Weak bonds (glycosidic C–O, benzylic, α-carbonyl C–C) explain peaks; strong bonds (aryl C–C, C–F) generate decoy masses that dilute discrimination | 100% (every molecule) | BDE lookup per bond (C–C/C–O/C–N × order/hybridisation table, ~15 entries); weight fragment masses or emit top-k low-energy breaks only | ~0 (table lookup at `mol_graph` time) | **1** |
| G2 | No neutral-loss ladder — only ±2 H shifts; −H₂O/−NH₃/−CO/−HCOOH never generated | NP MS/MS base peaks are serial H₂O losses (flavonoids, terpenoids, polyols); formate adducts lose HCOOH (46 Da) | OH-rich NPs dominant class; formate adducts 2.6% of test spectra directly affected; H₂O-loss peaks present in majority of NP spectra (literature: dominant NL in positive mode) | Append NL ladder to ion list: frag ± {H₂O, 2×H₂O, NH₃, CO, HCOOH} + proton (6 extra masses per frag, ~10% cost) | +10% frag time (~30 s) | **2** |
| G3 | Ring cleavage effectively blind — 1 break in a ring yields 1 component (intact mass); 2-break pairs rarely coordinate in same ring; `v7` excludes ring bonds entirely | 96.3% / 98% have ≥1 ring; 87.7% / 93% ≥2 rings. Fused/bridged NP ring systems fragment only via 2-in-same-ring or 3-break cascades | ~96% of candidates (NP-band) | Enumerate same-ring 2-break pairs explicitly (ring-bond index from RDKit, pairs within one SSSR ring) + depth-3 for ≤25-bond candidates only | +20–40% frag time, bounded by small-cand subset (~2–3 min) | **3** |
| G4 | No rearrangements: McLafferty, retro-Diels–Alder, sugar cross-ring (0,2X etc.) | McLafferty γ-H carbonyl motif 36% NP / 22% train; cyclohexene (rDA substrate) 12.6% NP / 1.1% train; glycoside substructure 3.2% NP / 0.5% train (SMARTS `C1OC(CO)C(O)C(O)C1O` — undercounts O-glycosides) | ~36% McL-capable, ~13% rDA-capable, ~3–10% glycosidic | MAGMa-style rules: McLafferty γ-H transfer mass (frag+1.0078 at γ-cleavage), rDA split masses for cyclohexene SMARTS hits, glycosidic-bond priority break + cross-ring pair; fire only on SMARTS match (cheap gate) | ~0 when gated (SMARTS prefilter, ~40% of cands get +2 masses) | **4** |
| G5 | 34-bond cap blinds large NPs — `nb > max_bonds` returns intact mass only | nb>34: **12.8% NP-band** vs 1.1% train-proxy; median nb 28 NP / 25 train. Cap hits exactly the glycosylated/large test tail | 12.8% of NP candidates score ~0 frag | Raise cap to 60 + subsample pairs (all singles, stratified pair sample, ring-priority) for nb>34; or precompute offline (`v7/precompute_frags.py` pattern) | sampling keeps ~flat; offline precompute 0 inference cost | **5** |
| G6 | No adduct-switching on fragments — all frags ionised ±proton by polarity (`PROTON_MASS if mode>0`) | [M+Na]+/[M+K]+ fragments often appear as [frag+H]+, [frag+Na]+, or post-Na-loss; formate-adduct fragments lose adduct; negative-mode charge retention differs | 1.8% Na/K + 2.6% formate spectra directly wrong adduct mass (100% miss on adduct-retaining frags); partial effect on all (which-fragment-keeps-charge unmodelled) | Ionise frag list under both H+ and precursor-adduct masses; take max explain (2× ion list, ~5% cost) | +5% | **6** |
| G7 | No charge-site retention model — every component assumed charged and explainable | Only the charge-retaining piece is observed; lite double-counts complementary neutrals as hits → inflates decoy explain scores, compresses truth/decoy gap | 100% (scoring bias), hurts isomer MRR (frag 0.28 vs analog 0.54) most | Basic-site retention prior (proton on N > carbonyl O > ether; Na on poly-O): keep masses of fragments containing best basic site, downweight others 0.5× | ~0 (one SMARTS/tag per frag) | **7** |
| G8 | H-shifts capped ±2 | rDA (±3 common), sugar cross-ring transfers, double-H rearrangements exceed ±2 | Subset of G4 (~5–15% of spectra show unexplained ±3/±4 satellites) | Extend to ±3 for SMARTS-gated cands only (rDA/sugar hits) | ~0 gated | **8** |
| G9 | Fixed `MZ_TOL=0.01 Da` tolerance; no instrument/CE adaptation | At test m/z 245–460, 0.01 Da = 22–40 ppm — 3–5× looser than precursor window (8.5 ppm) and timsTOF capability; admits false explains at low m/z, misses calibrated offset at high m/z. CE 20→60 changes fragmentation depth, lite is CE-blind | 100% (precision loss); all-CE queries blurred | `tol = max(5 ppm × m/z, 0.005)` + per-CE explain (score CE=60 spectra with depth-2, CE=20 with depth-1+NL) | ~0 | **9** |
| G10 | Intensity model = sqrt fraction only; `frag_scores` re-cleans peaks with entropy weighting OFF | sqrt underweights diagnostic low-abundance fragments; entropy-OFF cleaning disagrees with library channel cleaning (same peaks, different weights) | 100% (ranking compression) | Log-intensity weights + entropy weighting ON (match cell 6 `clean_spectrum` path); learn 2-param (weight, NL bonus) logistic on holdout | ~0 | **10** |

Notes:
- fork-034 vs v7 divergence: fork-034 enumerates ALL bonds (incl. ring/double) with
  `max_breaks=2, max_bonds=34`; v7 enumerates only single non-ring bonds and folds
  +H into fragment masses instead of H-shifted ion matching, and scores against a
  single adduct delta. Both share G1–G4/G6–G10; G3/G5 manifest oppositely
  (fork-034 wastes time on ring pairs that rejoin; v7 skips rings outright).
- `v7/validate_metfrag.py` (n=30, disjoint) was never run to completion in-repo
  (no results file); re-run it after fixes G1+G2 as the A/B gate (distrust
  deltas <0.007 per seed-noise protocol).

## Ranked fixes (expected MRR gain order)

1. **G1 BDE weights** — est. frag 0.259 → ~0.28, +0.002–0.003 LB. Zero CPU. Do first.
2. **G2 neutral-loss ladder** — est. +0.002–0.004 LB (dominant NP peaks currently
   unexplained). +30 s. Do second, A/B separately from G1.
3. **G3 same-ring 2-break + depth-3 for small cands** — est. +0.001–0.003 LB,
   concentrated in fused-ring NPs where analog is also weak (highest marginal value).
   +2–3 min. Do third.
4. **G4 rearrangement rules (gated)** — est. +0.001–0.002 LB on 36%/13% subsets.
   ~0 CPU. Do fourth.
5. **G5 cap lift via sampling/offline precompute** — converts 12.8% zero-scores to
   signal; est. +0.001–0.002 LB. Flat CPU with sampling.
6.–10. **G6/G7/G9/G10** — each est. ≤+0.001 LB alone; batch as one "scoring hygiene"
   submission (adduct-aware ionisation + retention prior + ppm tolerance +
   entropy-ON weights).

Total frag-channel aspiration: 0.259 → ~0.30–0.32 solo; stacked +0.005–0.010 LB
given 0.058 decorrelation. Budget: ~14 ms/cand baseline (~5 min/400 mols @MPool(4));
all fixes sum ≈ +30–50% frag time (≤8 min), inside the 9 h kernel.
