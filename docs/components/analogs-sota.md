# Analog/Neighborhood Retrieval SOTA — Beyond `sim^p * Tanimoto`

Companion to `analog-propagation.md` (do not repeat it: GNPS lineage, `analog_sim`
mechanics, and its §4–5 upgrade list are taken as given). This doc: 8 papers
downloaded to `docs/papers/analogs/` (all `file`-verified PDFs), per-paper verdicts
for OUR regime (Class-2 novel NPs, 9h offline CPU, train-only learning), then 3 NEW
analog mechanisms — none of them the baseline `max_j tan_ij * sim_j^4` copy, none of
them items 1–7 of `analog-propagation.md` §5.

## Sources (all local, all verified `PDF document`)

| File | Paper | Venue | Access route |
|---|---|---|---|
| `fbmn-Nothias2020.pdf` (4.5 MB) | Nothias et al., Feature-based MN | Nat Methods 2020 | WUR edepot (GREEN OA) |
| `iimn-Schmid2021.pdf` (3.3 MB) | Schmid et al., Ion-identity MN | Nat Commun 2021 | nature.com (Gold OA) |
| `ms2lda-vanderHooft2016.pdf` (4.5 MB) | van der Hooft et al., MS2LDA topic modeling | PNAS 2016 | Edinburgh Explorer (GREEN OA) |
| `molnetenhancer-Ernst2019.pdf` (4.5 MB) | Ernst et al., MolNetEnhancer | Metabolites 2019 | mdpi-res.com (Gold OA) |
| `spec2vec-Huber2021.pdf` (3.2 MB) | Huber et al., Spec2Vec | PLoS Comput Biol 2021 | PLoS (Gold OA) |
| `ms2deepscore-Huber2021.pdf` (2.3 MB) | Huber et al., MS2DeepScore | bioRxiv 2021.04.18.440324v1 (= J Cheminform 2021) | bioRxiv |
| `ms2query-deJonge2023.pdf` (2.5 MB) | de Jonge et al., MS2Query | Nat Commun 2023 | nature.com (Gold OA) |
| `flashentropy-Li2024.pdf` (2.8 MB) | Li et al., Flash entropy search | Nat Methods 2024 (eScholarship author copy) | eScholarship (GREEN OA) |
| `modcos-eval-Bittremieux2022.pdf` (2.6 MB) | Bittremieux et al., cosine vs modified-cosine vs neutral-loss | bioRxiv 2022.06.01.494370 (= JASMS 2022) | bioRxiv |

Download notes (repro): `pmc.ncbi.nlm.nih.gov/.../pdf/` serves a JS bot-check to
curl, and `europepmc.org/backend/ptpmcrender.fcgi` 403s — publisher/OA-repository
PDFs above are the working routes (Semantic Scholar `openAccessPdf` + EBI
`SRC:PPR` lookup to find them). Nat Methods versions of FBMN/FlashEntropy are
paywalled on nature.com; the GREEN copies are author manuscripts (same content).

## Per-paper verdicts (new material only — no GNPS-2016 recap)

1. **FBMN (Nothias 2020).** Two-step: LC-MS feature finding → one representative
   MS2 per feature → classical MN on features. Wins: resolves isomers with near-identical
   MS2 (7 vs 6 commendamide isomers found), adds RT/IM separation + relative
   quantification (R²-verified), kills chimeric spectra. VERDICT for us: retrieval
   logic unchanged (still cosine edges), so no direct MRR gain — but it VALIDATES our
   `build_rep` dedup direction and says: dedup key should ideally be
   (structure × adduct × CID-energy), not just inchikey14; replicate-collapse already
   does 80% of this. No action beyond current rep set.
2. **IIMN (Schmid 2021).** MS1 peak-shape correlation groups adducts/in-source
   fragments/multimers of one molecule BEFORE networking; collapses redundant nodes,
   connects ion species whose MS2 never matches. VERDICT: our `same-adduct prefilter +
   fallback` result (LB-neutral, §6 of companion) is the poor-man's IIMN and already
   captures most of it. The genuinely new bit: **in-source-fragment edges** — a
   query spectrum at mass M can match a library spectrum at mass M−loss as an
   in-source fragment, i.e. a second principled shift hypothesis beyond ΔM. Feeds
   Design B below. Standalone IIMN needs LC peak shapes (we have DDA spectra, not
   chromatograms) → not directly portable.
3. **MS2LDA (van der Hooft 2016).** LDA over spectra-as-documents,
   fragments+losses-as-words → ~300 Mass2Motifs/file, 30–40 biochemically
   characterized each, and 30 characterized motifs annotate ~3× as many molecules as
   library matching; per-spectrum motif count is small (mostly 1–2 of 300). VERDICT:
   the only paper here that retrieves on SUBSTRUCTURE overlap rather than
   whole-spectrum alignment — directly motivates Design A. Note the asymmetry it
   exploits: motif presence is far sparser and more interpretable than peak lists.
4. **MolNetEnhancer (Ernst 2019).** Integration layer, not a scorer: MN edges +
   NAP/DEREPLICATOR/VarQuest/SIRIUS annotations + MS2LDA motifs → ClassyFire
   chemical-class consensus per molecular family. VERDICT: no new similarity math,
   but two portable lessons: (a) family-level (not edge-level) consensus is the robust
   unit — supports voting over top-k analogs, not max-of-one; (b) VarQuest's "one
   modified residue" model is the peptide World's single-dM prior, same assumption
   class as ours. No direct feature; informs Design A/C evaluation (class-conditional).
5. **Spec2Vec (Huber 2021).** Word2Vec on co-occurring fragments/losses; 13k
   molecules, trained 15–50 iterations; correlates with Tanimoto better than cosine;
   ~60% of novel queries get top-10 Tanimoto>0.6 (random 1%); mean best Tanimoto
   >0.8 for >400 Da queries; 1030 queries × 76k refs = 140 s on i7 (≈0.14 s/query).
   VERDICT: best evidence that a cheap embedding arm beats whole-spectrum alignment
   on large NPs — but companion §5.4 already proposes exactly this arm. New residue:
   Spec2Vec's fine print — it wins where peak-match counts are high; on few-peak
   small molecules the min-match rule still zeroes pairs. So gate the Spec2Vec arm
   by query peak count, don't blend uniformly.
6. **MS2DeepScore (Huber 2021, bioRxiv v1).** Siamese net on 102k spectra/14k
   molecules directly predicting Tanimoto; RMSE ≈0.15 overall, ≈0.10 on the
   uncertainty-filtered subset (MC-dropout); beats both modified-cosine variants and
   Spec2Vec at retrieving related pairs; embeddings cluster chemically. VERDICT: the
   MC-dropout uncertainty head is the under-used idea — a *calibrated* predicted
   Tanimoto with a reject option is strictly more useful for propagation weighting
   than raw entropy sim. Directly motivates Design C (but ours is GBDT-free,
   logistic-calibrated, trainable in seconds on train.parquet — no GPU, no leakage
   surface beyond what we already have).
7. **MS2Query (de Jonge 2023).** RF over 5 features (Spec2Vec, MS2DeepScore,
   precursor-m/z terms, avg-MS2DeepScore and avg-Tanimoto over top-10 library hits);
   threshold 0.633 → 35% recall at mean Tanimoto 0.63 vs 0.45 for modified cosine at
   matched recall; 80 spectra/min laptop. VERDICT: two lessons, both portable without
   the RF: (a) the single most important feature is the **average over top-10
   library hits** — i.e. kNN-mean beats 1NN-max, contradicting our max-only `ap`;
   (b) the paper's "analog test set" (exact match removed) is the right offline
   protocol for tuning OUR channel — adopt it for Designs A–C ablations.
8. **Flash entropy (Li 2024).** Mathematically identical entropy similarity
   reformulated for indexing: >10,000× speedup, 1B spectra <2 s, median <1 ms/spectrum;
   formalizes identity / open (fragment-only) / neutral-loss / hybrid (per-peak
   fragment-OR-loss, fragment priority) searches; 25,000× faster than MatchMS on
   open/neutral-loss, 1,500× on hybrid. VERDICT: legitimizes wider windows — if we
   reimplement `search_shift` in the Flash formulation (or just precompute rep
   entropy tables once), the ±400 Da adaptive-K idea from companion §5.3 becomes
   ~free instead of ~2× cost. Engineering win, no scoring change.
9. **Modified-cosine eval (Bittremieux 2022 bioRxiv; 955k peptide + 10M
   small-molecule pairs).** Modified cosine strictly > neutral-loss-only > cosine in
   ALL settings — but performance depends on modification position/type and compound
   class; neutral-loss wins only in narrow niches (charge-remote / some peptide
   geometries); small-molecule ground-truth pairs had to be synthesized from GNPS
   refs (56.6%/70.9% stats). VERDICT: single-global-dM is the best *single* rule but
   provably non-uniform — the paper is the license for per-pair/per-class adaptive
   weighting (Design C) and for NOT trusting neutral-loss as a standalone channel
   (keep it as a GBDT feature at most, per companion §5.6).

`kNN-with-Tanimoto` theory note: no downloaded paper proves kNN consistency — that
is textbook (Cover–Hart 1967: 1NN error ≤ 2× Bayes; kNN-mean reduces variance at
the cost of bias). The PAPERS supply the domain premises that make kNN valid here:
Spec2Vec Fig 3's "oracle max-Tanimoto per percentile" curve is the Bayes-like ceiling
for retrieval; MS2Query's top-10-average feature (§2a above) is kNN-mean working in
practice; MS2DeepScore's RMSE-by-Tanimoto-bin plot shows the noise regime (low-sim
pairs are underestimated → max-pooling amplifies noise, averaging suppresses it).
Designs below assume: propagate over a k-neighborhood with calibrated weights, never
a single max edge.

Scaffold-hopping note: whole-spectrum shifted alignment CANNOT hop scaffolds by
construction (needs shared peaks in one global frame); modcos-eval §9 shows exactly
where it degrades (modification at fragmentation-directing sites, distributed edits).
Hopping evidence in these papers comes only from learned/substructure views:
Spec2Vec co-occurrence embeddings (>400 Da wins), MS2DeepScore direct-Tanimoto
prediction, MS2LDA motifs. Design A is our hopping mechanism; B/C are precision
mechanisms for the non-hop majority.

## NEW designs (ranked; all offline-CPU, train.parquet-only)

### D1. Motif-mediated propagation (substructure channel) — MOST PROMISING
Replaces whole-spectrum `sim` with motif-overlap for one additional vote per
candidate. Offline: mine K≈256 motifs from train.parquet rep spectra — LDA
(`sklearn.decomposition.LatentDirichletAllocation`, documents = spectra, vocab =
binned fragment m/z + top neutral losses) or cheaper NMF on the same count matrix
(~10 min CPU, once). Each motif k has a fragment distribution φ_k; score motif
presence in a spectrum by Σ_{peaks} φ_k(mz)·I (sparse dot, top-256 peaks).
Query motif vector q, analog motif vector a_j (precomputed for all reps once).
Motif support: `m_j = cosine(q, a_j)` (sparse, ~µs/pair).
Propagation (NEW — no sim^p anywhere):
`ap_motif(c) = max_j tan(c, analog_j) · m_j^2 · idf_j`,
`idf_j = mean_k q_k·a_jk·log(N/df_k)` (rare shared motifs count more; df from train).
Keep also `mean_top3` variant (MS2Query lesson: mean > max).
Why new: baseline and companion §5 all score EDGES between full spectra; this scores
SHARED SUBSTRUCTURES, so it fires on scaffold hops and multi-edit congeners where
`max(direct, shifted)` is ~0 but 1–2 motifs (hexose, adenine, prenyl, N-acyl amide…)
coincide. MS2LDA's 3×-annotation result is the existence proof.
Predicted effect: +0.004–0.009 Class-2 MRR (largest single-channel headroom: it
rescues queries where the current channel emits ~0). Feasibility: LDA/NMF once
offline; per-query cost ≈ top-100 analogs × 256-dim sparse dots — negligible next
to numba search. Leakage-safe (unsupervised, train spectra only). Ablation: remove
exact-match analogs (MS2Query protocol) and sweep K ∈ {128, 256, 512}.

### D2. Cross-spectrum shift-consensus voting (multi-shift without assignment)
Baseline max-pools analog sims over a molecule's spectra; companion §5.1 upgrades the
PAIRWISE assignment. D2 instead votes ACROSS the query's spectra (multi-adduct/CID
molecules are the norm): for analog j with per-spectrum best shifts δ_{j,s} and sims
s_{j,s}, define consensus `C_j = max_δ Σ_s s_{j,s}·1[|δ_{j,s}−δ| ≤ tol]` (tol = 15 mDa;
IIMN's in-source-fragment insight adds a second legal hypothesis family
δ′ = δ − common_loss, searched in the same vote). Weight:
`w_j = (Σ_s s_{j,s}·1[δ-agree]) / (Σ_s s_{j,s})` (fraction of spectral evidence behind
one shift story), feature `ap_cons(c) = max_j tan(c,analog_j)·w_j·s̄_j`.
Why new: a false-positive analog matches ONE spectrum at a random δ; a true congener
matches SEVERAL spectra at the SAME δ (same physical modification). Nothing in the
baseline or §5 uses cross-spectrum δ-agreement. Predicted effect: +0.002–0.006
Class-2 MRR, mostly precision (kills chance alignments p=4 currently keeps when sim
is spuriously high — the few-peak failure mode). Feasibility: zero extra spectral
comparisons — recomputes over stored per-spectrum (sim, δ) pairs; minutes of CPU.
Ablation: molecules with ≥3 spectra should show 2–3× the delta of singletons.

### D3. Learned propagation calibration (logistic, not another power)
Replace the hand-tuned `sim^4` gate with a fitted per-pair reliability:
`w_j = σ(α·s_j + β·min(1, n_j/6) + γ·1[δ_j ∈ COMMON] + λ·1[same adduct] + μ·log N_peaks_q)`,
COMMON = {14.0156, 15.9949, 18.0106, 28.0313, 42.0106, 68.0626, 162.0528, …}
(±10 mDa), n_j = matched-peak count. Fit (α,β,γ,λ,μ) by L2 logistic regression on
train.parquet Class-2-style pairs (label = 1[Tanimoto(analog,true) > 0.6]) —
seconds on CPU. Feature `ap_cal(c) = max_j tan(c,analog_j)·w_j` (+ top-3-mean twin).
Why new: MS2Query/MS2DeepScore prove learned combination beats any fixed
threshold/power, but both need embeddings/RF machinery; this is the minimal
calibrated version — it LEARNS the shift-tolerance (γ per-shift-type in the
extension: one coefficient per common loss) instead of asserting p=4. Companion §5.2
proposes the same raw signals as GBDT inputs; D3 differs by fitting the gate BEFORE
propagation, so the max/mean aggregation itself becomes noise-aware (modcos-eval §9:
non-uniform reliability is exactly what a global power cannot express).
Predicted effect: +0.003–0.007 Class-2 MRR. Feasibility: trivial; strictly additive
columns, no retrieval change, no leakage (labels from train structures only).
Ablation: drop each term; expect β (GNPS ≥6 rule) and γ (common-dM) to carry it.

Runner-up (not fully specified): Tversky-asymmetric propagation
`ap_tve(c) = max_j tversky(c, analog_j; α=0.3, β=0.7)` rewarding analog
substructure contained in the candidate — the cheapest scaffold-hop bias, one C8
column. Predicted +0.001–0.003; do after D1–D3.

## Build order
D3 first (hours, no retrieval change) → D2 (reuses stored per-spectrum pairs) →
D1 (needs LDA/NMF mining + rep motif index) → Flash-entropy reimplementation of
`search_shift` as the cost umbrella for ±400 Da. Evaluate each on the MS2Query-style
analog-only split (exact-match analogs removed) before trusting LB.
