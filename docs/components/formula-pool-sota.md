# Formula + Pool SOTA: 10 papers, verdicts, and 3 new designs

Branch: feat/domain-research | Date: 2026-09-24 | LB 0.155
Scope: (A) formula annotation from MS/MS, (B) candidate-pool design for novel NPs.
Goes beyond `constraints-and-literature.md` (MIST-CF, formula funnel) and `limits-analysis.md`
(pool recall, slot math). Papers on disk: `docs/papers/formula-pool/`
(2 bioRxiv PDFs + 8 PMC full-texts via EBI; `file`-verified; `.txt` = tag-stripped reading copies).
Hard constraint driving everything below: **test rows carry NO MS1** — only `precursor_mz` +
MS2 (`ms2_mzs`, `ms2_normalized_intensities`, 1–9 spectra/molecule, median 230 peaks/spectrum).
Any method needing MS1 isotope patterns must be re-engineered for MS2-resident signal.

## A. Formula annotation — per-paper verdicts

### A1. SIRIUS isotope decomposition (Böcker et al., Bioinformatics 2009, PMC2639009)
Bayesian scoring of simulated vs measured isotope patterns; >90% correct sum formulas to
1000 Da on oTOF **given clean MS1 isotope patterns**. Verdict: core theory is directly reusable
(knapsack mass decomposition + pattern likelihood), but the input it assumes (MS1 M/M+1/M+2)
does not exist in our test. Do NOT port as-is; salvage the decomposition math for the
MS2-resident isotope idea (N1 below). Note it already warns: candidates explode with mass and
with elements beyond CHNOPS — our alphabet must stay tight (CHNOPS + halogens + Na/K for
adducts only).

### A2. Seven Golden Rules (Kind & Fiehn, BMC Bioinformatics 2007, PMC1851972)
Heuristic filters (element counts, LEWIS/SENIOR valence, H/C ratios, isotope-fit, mass defect)
cut billions of decompositions to ~623M plausible; on 6000 DrugBank/TSCA/DNP compounds the
correct formula is top hit at 80–99% **assuming 3 ppm + 5% isotope-ratio error**. Verdict: still
the cheapest first-pass filter we have. Copy the SENIOR/LEWIS + element-ratio gates into our
formula enumerator (zero GPU, kills absurd CHNOPS combos before any neural scorer sees them).
Their 3 ppm/5% assumption is optimistic for us (no MS1, timsTOF MS2 only) — treat as filter,
never as ranker.

### A3. Fragmentation trees reloaded / SIRIUS 3 (Böcker & Dührkop, J Cheminform 2016, PMC4736045)
Replaces heuristic tree scoring with Bayesian statistics; best tree ≈ most likely fragmentation
cascade; SIRIUS 3 beats v2 on both de-novo formula ID and DB search for analogs. Verdict: this
is the intellectual parent of MIST-CF's subformula labelling (already in our v1 plan). New take
for us: their result that **tree-alignment similarity detects analogs** justifies using
fragmentation-tree overlap (common fragments + root losses, cheap count version per ZODIAC
Eq.6) as a *pool-internal* similarity for diversity filling and for the cross-spectrum formula
intersection (N2). Do not reimplement full trees — MIST-CF already showed top-20 subformula
labels saturate accuracy.

### A4. ZODIAC (Ludwig et al., bioRxiv 10.1101/842740 → Nat Mach Intell 2020)
Reranks SIRIUS top-50 formulas with Gibbs sampling over a fragmentation-tree-similarity graph
of co-occurring compounds; problem is NP-complete (reduction from Clique), sampler gives ~25x
engineered speedup. Results: dendroides error 49.2%→3.0% (**16.2x**), SIRIUS 50.8%→ZODIAC
96.9% correct; score>0.9 ⇒ >96.5% correct while keeping 52–88% of compounds; discovered
C24H47BrNO8P (absent from PubChem/ChemSpider), validated 6 ways incl. M+2 fragment
simulation (Rockwood) and Br M+2 106% intensity signature. Two warnings that matter for us:
(1) all 6 residual dendroides errors were **H⁺/Na⁺ adduct confusions** (Δ=21.9819 Da vs
C2H-2 21.9843 Da — 2.4 mDa apart); (2) with known adducts, error drops 66x. Verdict: biggest
lesson is not Gibbs sampling (needs whole LC-MS runs; we have 400 isolated molecules) but
**adduct correctness dominates formula correctness**, and M+2-simulation validates halogens.
Directly motivates N2 (adduct voting) and the halogen branch of N1.

### A5. BUDDY (Xing et al., bioRxiv 10.1101/2022.08.03.502704 → Nat Methods 2023)
Bottom-up: decompose every fragment–neutral-loss pair against 3.5M curated subformulae,
stitch survivors into precursor candidates (SENIOR-filtered), rank with LightGBM MLR
(38 MS1+MS2 features), Platt-calibrate to FDR (AUC 0.985, Q-Q r>0.98). Results: candidate
space −42.8% avg (7.9x @400 m/z, 12.2x @600, **20.7x @800**); top-1 +30.1% avg over SIRIUS
(+68.7% on natural-product VF-NPL!); 93.0% for m/z<400 but only 63.8% ≥400; found 226
PubChem-absent formulae in ARUS at FDR<5%. Verdict: most actionable paper in this set.
Three portable pieces: (i) bottom-up fragment→precursor stitching beats top-down enumeration
— our formula ranker should generate candidates FROM explained fragments, not from mass
decomposition alone; (ii) LightGBM-on-38-features + Platt FDR is exactly the shape our
reranker should copy (and gives us the confidence estimator N3 needs); (iii) VF-NPL +68.7%
says bottom-up helps NPs most — our test domain. Caveat: needs MS1 isotopes for full power;
their no-MS/MS ablation (−34% on Chagas) proves MS/MS carries signal we can use alone.

### A6. Ion Identity Networking (Schmid et al., Nat Commun 2021, PMC8219731)
Groups [M+H]⁺/[M+Na]⁺/[M+NH4]⁺/in-source fragments of one molecule via chromatographic
peak-shape correlation; collapsing ion identities cut a demo network 43 nodes → 4 molecules
(−56%) and enables annotation propagation to unknowns. Verdict: we lack RT/peak shape, but
the abstraction transfers: **identity resolution = group ion species by a shared latent
(neutral mass) before ranking**. Our equivalent signal is exact neutral-mass agreement across
a molecule's spectra (<6 ppm, limits-analysis §2) plus shared subformula fragments — the
mechanism of N2. Also legitimizes treating Na⁺/K⁺/formate spectra as views of the same
neutral, not independent queries.

## B. Pool design for novel chemistry — per-paper verdicts

### B1. COCONUT (Sorokina et al., J Cheminform 2021, PMC7798278)
53 sources aggregated → 406,076 flat + 730,441 stereo NPs (Oct 2020). Verdict: confirms our
COCONUT-new pool (≈627K) is the largest open NP structure set, but also its limits: only 43%
in our 246–439 Da test band, median mass 431 vs test 328. It is a *known*-NP pool; truly novel
test answers are definitionally absent. Keep as recall layer 2, never as the whole pool.

### B2. LOTUS (Rutz et al., eLife 2022, PMC9135406)
750,000+ referenced structure–organism pairs on Wikidata (420 manually validated), i.e. the
largest structure↔species graph. Verdict: structures overlap COCONUT heavily — do NOT add as
another structure pool (dilution without recall). Its value is as a **prior**: organism/taxon-conditioned
candidate weighting. We have no organism labels for test molecules, so this stays a
future lever (e.g. mass-shift series consistent with known biotransformations), not v2 work.

### B3. NPAtlas 2.0 (van Santen et al., NAR 2022, PMC8728154)
32,552 microbial NPs (+8,128), standardized dereplication set with taxonomy + NPClassifier/
ClassyFire ontology. Verdict: highest-precision *dereplication* pool (microbial only, curated,
small). Predicted effect as an added pool layer: +1–3% recall on microbial-like test molecules
at near-zero ranking cost (small pool). Worth concatenating as pool layer 1.5 (train → NPAtlas →
COCONUT-new), with a source prior feature in the reranker.

### B4. MAYGEN (Yirik et al., J Cheminform 2021, PMC8254276)
Orderly-generation constitutional-isomer enumerator: exhaustive, isomorphism-free, 47x faster
than PMG (3x slower than closed MOLGEN), pure Java. Verdict: full isomer enumeration is
intractable per molecule (millions at 328 Da), but MAYGEN proves *constrained* enumeration is
fast. Do not enumerate blindly — enumerate **one-reaction-step analogs of top retrieved
scaffolds** (N3a). The orderly-generation principle also warns: dedup by canonical key
(InChIKey14, already mandatory) is what keeps enumerated pools from wasting slots.

### Dilution theory (computed, no paper needed)
limits-analysis §2: median combined pool = 188 structures / 9 formulae at ±20 ppm. A
PubChem-scale pool (10⁸) at the same tolerance would put ~10⁴ same-mass candidates per
molecule (scaling from 188 @ 7×10⁵) — rank-1 becomes a lottery even for a perfect scorer,
while formula collapse still only buys ~14x. Conclusion: pool *relevance* (NP-focused,
mass-matched) beats pool *size* by orders of magnitude; every pool addition must be gated on
mass-band overlap (COCONUT fails 57% of the time here) and on the recall estimator (N3b).

## C. Three new designs (ranked by expected MRR gain / implementation cost)

### N1. MS2-resident isotope rerank (no MS1) — BEST BET, +0.008–0.02 MRR
Observation: test spectra are dense (median 230 peaks vs train 42). Wide-window timsTOF
isolation co-captures M+1/M+2 satellites of fragments (ZODIAC mice-stool saw exactly this;
BUDDY has a deisotoping note for the same reason). Design: for each top-K formula candidate,
(i) predict which intense fragments contain Br/Cl/S (M+2 ≈ 98%/32%/4.4% signatures à la
ZODIAC Fig.3: Br M+2 at 106% of mono), and scan the query MS2 for +1.9979/+0.9995 Da
satellites of the top-20 explained fragments; (ii) score precursor-region satellites the same
way; (iii) add satellite-match log-likelihood + halogen-vote as features to the formula LightGBM
(BUDDY-style MLR). No MS1, pure NumPy, per-molecule cost trivial. Why new: SIRIUS/BUDDY
both consume MS1 isotopes; nobody scores *fragment* isotope satellites inside MS2 as a
formula reranker. Predicted: formula top-1 +3–6 pts on dense spectra → pool shrinks 14x
correctly more often → MRR +0.008–0.02. Failure mode: sparse spectra (<50 peaks) get no
satellites — gate the feature on peak count.

### N2. Cross-spectrum adduct voting + formula intersection — +0.005–0.01 MRR, cheapest
Observation: 79/400 molecules have ± dual spectra; masses agree <6 ppm across adducts;
ZODIAC proved adduct errors dominate formula errors (H⁺/Na⁺ 2.4 mDa trap). Design: build a
per-molecule adduct graph — nodes = (spectrum, adduct hypothesis from the 7-offset list),
edges when implied neutrals agree <10 ppm; take the max-consistent clique as the neutral mass
(IIN collapse without RT). Then intersect: keep only formulas that explain ≥1 fragment in
*every* spectrum of the molecule (BUDDY stitching, but the candidate must survive all views;
rare K⁺/Cl⁻/NH4⁺ spectra get rescued by the molecule's H⁺/H⁻ spectra instead of ranking
alone). Why new: all papers are per-spectrum; our 1–9 spectra/molecule structure is an unused
edge (constraints doc notes it, no paper does it). Predicted: rescues most of the ~8 rare-adduct
molecules (each rank-1 rescue = +0.0025) + tightens formula on dual-mode 79 → +0.005–0.01.
Cost: an afternoon of NumPy.

### N3. Focused-enumeration fillers gated by a pool-recall estimator — +0.005–0.015 MRR
(a) Enumerate one-step NP analogs of the top-3 retrieved scaffolds per molecule using the
ZODIAC biotransformation list as operators (glycosyl ±C6H10O5/±C6H10O4/±C5H8O4, methyl
±CH2, hydroxyl ±O, acyl swaps, Cl/Br substitution, dehydro ±H2): RDKit templates, cap ~200/
molecule, keep only ±20 ppm + SENIOR-valid (MAYGEN lesson: constrain, then dedup by
Key14). Targets the likeliest dark-matter mode: novel analog of a known NP scaffold
(cf. BUDDY's glycine-bile-acid and FAA discoveries — conjugation + homologation series).
(b) Gate with a calibrated P(truth-in-pool) = Platt-scaled retrieval margin (top1−top5 score
gap + formula confidence from N1/N2, BUDDY §FDR recipe); fillers take ONLY slots 16–25
and ONLY when P < threshold (limits-analysis slot math: those slots worth 0.06→0.04 each,
so misfiring costs ~nothing, hitting pays). Why new: MAYGEN enumerates blindly, MS2Mol
generates unconditionally; nobody routes enumeration by a calibrated recall estimate into
trailing slots. Predicted: +5–10% recall on novel-analog molecules → MRR +0.005–0.015
(upper end if >15% of test is analog-dark-matter). Build the estimator first (needed anyway
for slot policy); enumeration second.

## Build order
1. N2 (half-day, pure logic, rescues rare adducts) → 2. N1 (formula rerank features, needs
labeled dev spectra with dense MS2) → 3. N3b estimator (reuses N1/N2 confidences) → 4. N3a
enumeration (only if N3b shows a confident-dark subset). Validate all on InChIKey14-disjoint
splits — MassSpecGym proved Tanimoto>0.85 leakage inflates local scores.
