# Component: Spectral Similarity SOTA (beyond Li-2021)

Status: research doc. Do NOT repeat `library-search.md` (Li-2021 entropy port, fork-034 constants,
why our cosine-surviving/entropy-dying validation happened). This doc covers what the literature
says *beyond* that baseline, and designs new kernels that are not copies of Li-2021.

Corpus: `docs/papers/similarity/` — 6 verified PDFs (`file`-checked) + 2 full texts read via
PMC (Europe PMC PDF endpoint is Cloudflare-blocked from this network, same as ChemRxiv):

| File / source | Paper | Read |
|---|---|---|
| `spec2vec-plos2021.pdf` | Huber et al., PLoS Comp Biol 2021 (Spec2Vec) | full |
| `ms2deepscore-jcheminf2021.pdf` | Huber et al., J Cheminform 2021 (MS2DeepScore) | skimmed (abstract, Figs 2–3, MC-dropout §) |
| `ms2query-natcommun2023.pdf` | de Jonge et al., Nat Commun 2023 (MS2Query) | skimmed (method, speed, feature-importance §§) |
| `ms2deepscore2-biorxiv2024.pdf` | de Jonge et al., bioRxiv 2024.03.25 (MS2DeepScore 2.0, cross-ionization) | full |
| `entropy-li-natmethods2021.pdf` | Li et al., Nat Methods 2021 (eScholarship author MS) | skimmed (entropy concept, S<3 rationale, benchmark, FDR tests) |
| `wasserstein-wabi2018.pdf` | Majewski et al., WABI 2018 (Wasserstein/EMD for spectra) | full |
| PMC11511675 (webfetch) | Li & Fiehn, Nat Methods 2023 (Flash entropy search) | full text |
| PMC5841953 (webfetch) | Moorthy et al., Anal Chem 2017 (Hybrid Similarity Search) | full text |

Two more papers were NOT retrieved, cited second-hand (hedged below): Bittremieux et al.,
JASMS 2022 (cosine / modified-cosine / neutral-loss comparison, via Flash ref 13) and
Turkina et al., Anal Chem 2026 (20-metric benchmark, via abstract only).

## 1. Per-paper verdicts (one paragraph each)

**Spec2Vec (Huber 2021).** Word2Vec on spectral "documents" (peaks + neutral losses as words,
window = whole spectrum, CBOW): spectrum vector = sqrt-intensity-weighted sum of word vectors,
similarity = cosine between spectrum vectors. On 12,797 GNPS spectra: top-0.1% Spec2Vec pairs
have higher mean Tanimoto than top-0.1% cosine/modified-cosine pairs; library matching (1 ppm
precursor prefilter, 1000 queries) 88% accuracy at better TPR/FPR than cosine; 1030 unknown
queries vs 76k library in 140 s CPU (0.14 s/query, embeddings precomputable). Verdict: the
strongest *analog-task* (Tanimoto-proxy) unsupervised score and the fastest prefilter architecture
(fixed-length embeddings), but it optimizes structural relatedness, not identity — for Channel 1
(same-mass, same-adduct, answer-in-library) it is the wrong objective, and it needs a large
training corpus covering the peak vocabulary (their missing-fraction guard: reject spectra with
>5% unknown weighted vocabulary). Relevance to us: embedding-prefilter pattern + sqrt-intensity
spectrum-vector construction, not the kernel itself.

**MS2DeepScore (Huber 2021).** Supervised Siamese net: binned spectra → 200-d embeddings,
trained to regress Tanimoto (Daylight fingerprints won over Morgan/Dice on label-balance
grounds) on >100k spectra / ~15k compounds; RMSE ≈ 0.15 overall, ≈ 0.10 on uncertainty-filtered
subset (Monte-Carlo dropout ensembles + IQR-over-replicates filtering). Beats Spec2Vec and
modified cosine on precision/recall of Tanimoto>0.6 pairs (3601-spectra test). Verdict: strictly an
analog-task tool (predicts fingerprint similarity, blind to exact identity by construction); GPU
training, binning loses resolution, and MC-dropout uncertainty was later judged subpar by the
authors themselves (see 2.0). Not a Channel-1 kernel; candidate only as Channel-2 analog
pre-ranker with embeddings precomputed offline.

**MS2Query (de Jonge 2023).** Production wrapper around the two learned scores: MS2DeepScore
preselection → 5-feature random forest re-rank (dominant feature: *average MS2DeepScore over
multiple chemically similar library molecules* — first method to exploit "if library molecule A is
a good analog, its near-neighbor B should score high too"), plus precursor-m/z features; cosine /
modified cosine tested as extra features and *rejected* (no gain). 5987 queries vs 302k library, no
precursor prefilter: 74 min laptop CPU (80 spectra/min) vs 9.4 h for modified cosine with 100 Da
prefilter. Useful analogs (mean Tanimoto 0.63) for ~35% of spectra. Verdict: two transferable
ideas — (a) neighbor-consensus re-ranking (score a candidate by the scores of its structural
neighbors, not just itself), directly portable to our ranker as features; (b) learned-score
preselection makes un-prefiltered search tractable. The 5-feature RF pattern is the closest
published analog of our ranker-over-channels design.

**MS2DeepScore 2.0 (de Jonge 2024/25, bioRxiv).** Single dual-polarity model (precursor m/z +
ionization mode as metadata inputs; adduct also helped but was excluded from the default model
as unreliably annotated; instrument-type one-hot did NOT help), 500-d embeddings, single
10k-node layer, balanced Tanimoto-bin + per-molecule pair sampling (<15% frequency skew),
InceptionTime "Embedding Evaluator" predicting per-spectrum MSE (replaces MC dropout; MSE
correlates with #fragments, precursor m/z, intensity). Cross-polarity example: rutin +/− spectra
with zero overlapping fragments score 0.7 predicted Tanimoto vs 0.3 modified cosine / 0.0
cosine. Verdict: this is the adduct/instrument-awareness answer from the literature — metadata
helps, *except* instrument identity (one-hot instrument hurt-or-nothing: instrument effects are
absorbed better by replicate diversity than by labels, supporting our max-over-replicates
aggregation over instrument-gated features). Directly relevant: precursor-m/z-as-input validates
our neutral-window + adduct-gate design; per-spectrum MSE evaluator validates adding
spectrum-quality features to the ranker.

**Entropy, Li 2021 (re-read for what library-search.md omits).** Load-bearing details beyond the
formula: (i) the S<3 weight `w=0.25+0.25S` exists because near-identical molecules (ADP/ATP)
share the base peak (adenine m/z 136.062) and differ only in low-abundant ions — weighting is
an *intensity-weighting theory* statement, not a hack; (ii) 43-algorithm benchmark incl. weighted
dot-product variants all lost (AUC 0.958 entropy); (iii) noise-robustness and FDR<10% at 0.75 on
37k natural-product spectra; (iv) precursor-ion exclusion + RT (RetiP) combination is what made
gut-metabolome annotation work — i.e., the kernel alone was never the full system. Verdict:
confirms port-verbatim-first; the ADP/ATP rationale is the theoretical hook for Design 3 below
(downweight ubiquitous intense fragments, upweight rare discriminative ones).

**Flash entropy (Li 2023).** Mathematically identical reformulation of entropy similarity needing
*only matched-peak intensities* (mismatches enter purely via the sum-to-0.5 normalization) over
a globally m/z-sorted ion table — 25,000× faster than MatchMS open search, 5–10× on identity
search, O(s+p) memory. Crucially, it generalizes entropy to four search modes the original could
not do: identity, open (no precursor constraint), neutral-loss (precursor−fragment transform),
and **hybrid** (each query ion matches either a fragment or a neutral loss, fragments first, each
ion used once). Cleaning used: drop m/z > precursor−1.6, 1% floor, entropy weights, sum→0.5.
Verdict: two takeaways — (a) the matched-only reformulation is a free CPU win for any entropy
kernel we write (skip pairs with no common ions); (b) hybrid entropy search is the sanctioned
form of shift-tolerant matching *beyond constant-dM*: per-peak choice of fragment vs NL match
instead of one global shift.

**Hybrid Similarity Search (Moorthy 2017, NIST).** Precursor-mass-derived Δm shifts library
peaks (Biemann shift); each query peak matches direct or shifted library peak with intensity
*apportioned to conserve total abundance and maximize the match factor* (`hMF(Q,L,Δm) =
sMF(Q,H)`); single-modification pairs jump (caffeine→8-chlorocaffeine 260→865; fentanyl hit
lists: 10/10 FRCs >800 hybrid vs 8/10, 3>800 simple) while double modifications stay low —
i.e., it is a *single-inert-moiety* detector with a built-in degree-of-modification gradient. Noise
degrades hybrid ≈ simple (±20–42 pts). Score-lift-vs-mass `Ωn` even estimates unknown nominal
mass. Verdict: the principled ancestor of "shift-tolerant alignment beyond constant-dM", but
EI-era (unit mass, C=999 Stein composite `sMF`, eq.1 — itself the intensity-weighting prior art:
empirically derived, abundance-product form). For our HR-MS/MS setting the portable pieces
are abundance-conserving apportionment and the fragment-vs-shifted competition per peak
(Flash hybrid does this for entropy; Design 1 does it inside our window).

**Wasserstein/EMD (Majewski 2018).** Spectra as probability measures; W1 = minimal
intensity×|Δm/z| transport cost, on R computed in O(n+m) by a two-pointer CDF walk
(Algorithm 1 — same shape as our entropy two-pointer walk). MS1 W1 ≈ |Δmass| (ρ=0.89);
MS2 relative-W1 (÷ mass product) correlates with Tanimoto (−0.41) better than Jaccard (0.22).
Caveats stated by the authors: all mass must be explained → chemically noisy spectra globally
rewire the transport plan; mass-difference≡chemical-difference assumption needs the
mass-product correction; best on similar-mass, similar-precursor-fraction spectra. Application
shown is deconvolution-as-LP, not retrieval. Verdict: EMD is a *calibration-error-tolerant*
metric (small m/z errors shift mass slightly instead of breaking binary matches), implementable
in our numba idiom — but unusable raw (noise must be allowed to stay unmatched). Design 2
is the fenced version.

## 2. New similarity designs (ranked, all offline-CPU numba)

All three reuse Channel-1 context (±8.5 ppm window, same-adduct gate, clean floor 0.002 /
top-256 / power 1.0 / S<3 weights, tol 0.01) and change only the kernel. None is Li-2021 or the
dead matched-only variant (unmatched peaks always penalize, via normalization or explicit
terms). Predicted deltas are on **exact-InChIKey14 MRR@25** from our 0.155 baseline.

### Design A (prototype FIRST): dual-channel entropy — direct + neutral-loss, fragment-priority
*Mechanism.* For each query/candidate pair compute two entropy similarities with the identical
kernel: `S_dir` on fragment m/z, `S_nl` on neutral-loss m/z (precursor − fragment, requires both
precursor masses; fall back to `S_dir` when either is NaN). Final `S = max(S_dir, S_nl)` for
retrieval, and emit both plus the gap as ranker features (mirrors MS2Query's "learned score +
re-rank features" split, and Flash's hybrid with fragment priority). Rationale: adduct-gated
identity pairs share fragments (S_dir saturates, as now); cross-energy/instrument pairs and
single-moiety analogs share neutral losses even when fragments shift — the channel our current
single kernel is blind to. Cost: 2× kernel calls, both matched-only short-circuit (Flash eq. 5).
*Predicted effect:* **+0.005 to +0.015.** Near-zero on Class-1 (already saturated), all gain on
Class-2-via-analog-ranker and on cross-instrument Class-1 rescues. Prototype = one new numba
kernel + 3 ranker features; no index changes.

### Design B (prototype SECOND): library-PMI reweighted matching (Spec2Vec without training)
*Mechanism.* One offline pass over train builds a table over 0.01-Da bins: `pmi[i] =
log(P(co-occur with query context)/P(i))`, clipped at 0 — operationally, per-bin IDF:
bins seen in >X% of library spectra (water loss 18.01, adenine 136.06, ubiquitous low-mass
fragments) get weight →0.2, rare bins keep 1.0. Fold into the entropy mixture: matched-pair
contribution `(qp+cp)` becomes `(qp+cp)·w_bin`, unmatched singletons unchanged (penalty
preserved). This is the ADP/ATP lesson from Li-2021 made quantitative, and Spec2Vec's
co-occurrence insight in closed form (no Word2Vec, no vocabulary gaps, no GPU).
*Formula sketch:* `SAB* = H(mix with matched mass scaled by w_bin)`, similarity
`1 − (2·SAB* − SA − SB)/ln4`, with `w_bin = clip(log(N/df_bin)/log(N), 0.2, 1)`.
*Predicted effect:* **+0.005 to +0.015**, concentrated in Class-1 rank-1 precision (fewer
base-peak-coincidence false hits among the median ~52 window candidates). Risk: needs a
df-table ablation (rare-but-noisy bins); prototype = build table + one kernel flag.

### Design C (prototype THIRD): windowed EMD with entropy gate (calibration-tolerant kernel)
*Mechanism.* 1-D W1 via the WABI two-pointer CDF walk, but transport allowed only within
±0.05 Da per unit mass (beyond that, mass stays unmatched and penalizes through the entropy term): `S = exp(−W_win/τ)·S_ent^α`, τ ≈ 0.02·precursor_m/z,
α ≈ 0.5, all grid-searched. Rationale: converts our brittle 0.01-Da binary match into a graded
penalty (their +1.4 ppm timsTOF offset note; IT/low-res tails), while the entropy gate keeps the
Wasserstein "explain everything" pathology fenced. *Predicted effect:* **+0.002 to +0.008**,
mostly low-res/instrument-edge molecules; value is partly diagnostic (entropy–EMD
disagreement flags miscalibration). Prototype after A/B; needs τ/α/window grid.

*Ranking: A > B > C* on expected MRR per implementation hour (A: hours, reuses kernel;
B: hours + one table; C: days + grid). A and B stack (different failure modes: missing-match
vs false-match); C is orthogonal insurance.

## 3. What to prototype first (concrete order)

1. **Design A, retrieval-only max** (half day): add NL transform + `max(S_dir, S_nl)` to the
   Channel-1 scorer; validate Class-1 retention ≈ 1.0 and Class-2 feature lift before touching
   the ranker. Adopt Flash's matched-only short-circuit at the same time (free speedup).
2. **Design A features** (`S_dir, S_nl, gap`, agreement-with-adduct-gate): retrain ranker.
3. **Design B df-table**: build bin-df over train, ablate floor {0.2, 0.3, 0.5}, check Class-1
   top-1 movement only (it should never move Class-2).
4. **Design C grid** only if A+B leave instrument-edge failures.
5. **Do NOT**: Spec2Vec/MS2DeepScore inference for Channel 1 (wrong objective + GPU +
   vocabulary risk; library-search.md §5.7 stands). Neighbor-consensus re-ranking à la
   MS2Query is ranker work, not kernel work — file under ranker-calibration.

## 4. Intensity-weighting / adduct / instrument notes (for the record)

- *Intensity weighting:* Stein composite (`sMF`, Hybrid eq. 1) → sqrt transforms in cosine
  flavors (Spec2Vec S2: tolerated, never decisive) → Li S<3 weights (linear power won their
  ablation 0.919 vs 0.895, matching fork-034's INT_POWER=1.0). Consensus: keep power 1.0,
  put selectivity in per-bin weights (Design B), not global exponents.
- *Adduct-aware:* MS2DeepScore-2.0 shows adduct metadata helps *when reliably annotated*;
  ours often isn't → keep physical adduct-gate + 30-entry table (library-search.md §5.3),
  don't feed adducts to learned models.
- *Instrument-aware:* MS2DeepScore-2.0's instrument one-hot did not help; max-over-replicates
  already absorbs CE/instrument spread. No instrument kernel work justified.
