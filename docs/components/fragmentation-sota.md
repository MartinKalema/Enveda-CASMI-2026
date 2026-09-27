# Component: In-Silico Fragmentation SOTA (beyond MetFrag-lite)

Status: research doc. Do NOT repeat `pool-frag-validation.md` §1B/§3b (MetFrag-lite
mechanics, CFM-ID/ICEBERG/FraGNNet one-line verdicts, "offline-distill" direction).
This doc goes deeper and wider: full per-paper teardowns and NEW scorer designs that
beat 1–2-bond uniform-cost BFS inside the 9 h offline-CPU kernel.

Corpus: `docs/papers/fragmentation/` — 8 arXiv PDFs (`file`-verified) + 5 PMC full
texts (JATS XML/BioC; PMC PDF endpoints are bot-walled from this network) + `txt/`
extracts. One paper NOT retrieved, cited second-hand (hedged): Ridder et al., MAGMa,
Mass Spectrom 2014 (via ICEBERG/FraGNNet/MetFrag descriptions).

| File / source | Paper | Read |
|---|---|---|
| `arxiv-2304.13136.pdf` | Goldman et al., ICEBERG (NeurIPS 2023 workshop → ICML-adjacent 2024) | full |
| `arxiv-2404.02360.pdf` | Young et al., FraGNNet (TMLR 08/2025) | full |
| `arxiv-2303.06470.pdf` | Goldman et al., SCARF (NeurIPS 2023) | full (method, Tables 1–2, §4.3) |
| `arxiv-2301.11419.pdf` | Murphy et al., GrAFF-MS (ICML 2023) | full (method, Tables 1, §6.3–6.4) |
| `arxiv-1811.08545.pdf` | Wei et al., NEIMS (ACS Cent Sci 2019) | full |
| `arxiv-2111.04824.pdf` | Young et al., MassFormer (Nat Mach Intell-adjacent 2023) | skimmed (method, Fig 2, availability) |
| `arxiv-1412.1929.pdf` | Dührkop/Böcker, Fragmentation Trees Reloaded (J Biomed Sem 2015) | skimmed (scoring, Fig 4, datasets) |
| `arxiv-2410.23326.pdf` | Bushuiev et al., MassSpecGym (NeurIPS 2024) | skimmed (challenges, Tables retrieval/simulation) |
| `PMC4086103.xml` | Allen et al., CFM-ID web server (NAR 2014) | full |
| `PMC9064193.xml` | Wang et al., CFM-ID 4.0 (Anal Chem 2021) | full (method, Tables 1–3) |
| `PMC4732001.xml` | Ruttkies et al., MetFrag relaunched (J Cheminform 2016) | full (method, Table 1, runtime) |
| `PMC11889238.xml` | Nowatzky et al., FIORA (Nat Commun 2025) | full (method, Table 1, retrieval §) |
| `PMC12245663.xml` | Pérez-Ribera et al., SingleFrag (Brief Bioinform 2025) | full (method, Figs 3–4) |

## 1. Per-paper verdicts

**CFM-ID series (Allen 2014/2015 → Djoumbou-Feunang 3.0 → Wang 4.0).** Exact method:
probabilistic generative model over a fragmentation graph. SE-CFM learns per-transition
breakage probabilities by EM over (molecule, multi-energy spectra) pairs; inference
enumerates fragments breadth-first, recursing only into high-probability nodes, and
predicts spectra from marginal peak probabilities (top 80% intensity, 5–30 peaks).
4.0 adds (i) molecular-topology features, (ii) sequential ring cleavage (a 2-bond ring
cut = two ordered 1-bond transitions, sharing parameters with acyclic breaks), and
(iii) SMIRKS rule-based predictor for 11 classes × 3 adduct types (acylcarnitines,
flavonoids, …). Training: METLIN QTOF, 12,165 [M+H]+ / 6,120 [M−H]− spectra
(4,055/2,040 structures) at 10/20/40 eV. Metrics (4.0 vs 2.0/3.0, [M+H]+ overall):
Dice 0.30→0.37, dot-product 0.30→0.38, Precision 0.26→0.50 at Recall 0.44→0.37;
ablation: topology ~65%, ring features ~35%, rules ~1% of the gain. CASMI-style ID:
147/208 Top-1 in-silico-only (vs 120 CFM 2.0, 138 SIRIUS4+CSI:FingerID, 146 MS-FINDER).
Code: web service + Docker (`wishartlab/cfmid`), CLI ex-SourceForge; license is
academic/free-use — VERIFY before redistributing weights. Offline verdict: **distill
only**. Too slow for the kernel by ~100× (SCARF bench: 1,115 s/100 mols CPU;
MetFrag paper: CFM-ID 209 h vs MetFrag 75 min on 84 queries) and EM retraining takes
~3 months/64 cores on 300k spectra (per Murphy). Precompute predicted spectra or
transition tables per pool structure offline, ship `.npy`.

**ICEBERG (Goldman 2024).** Exact method: two modules. Generate: gated-GNN encodes
(root mol, current fragment, formula encodings, #bonds-broken, adduct one-hot) and
predicts per-ATOM breakage probability (MAGMa convention: atom removal ⇒ every event
changes heavy-atom composition; rings handled naturally since no explicit bond-pair
enumeration); trained on MAGMa-enumerated "canonical DAGs" (depth ≤3, WL-hash dedup,
greedy parent-cover pruning) with BCE loss, top-100 fragments kept. Score: Set
Transformer over generated fragments predicts intensities at {+0H, ±1…±6H} shifts per
fragment (McLafferty-style H-rearrangements as learned mass shifts, not rules),
attention-pooled into 0.1 Da bins, trained on cosine similarity. Training: NPLIB1
(10,709 spectra/8,553 structures, NP-heavy, mean 413 Da) + NIST20 (35,129/24,403,
mean 317 Da), 90/10 structure-disjoint. Metrics: cosine 0.627 vs 0.568 MassFormer
(+10%) on NPLIB1; tied SCARF on NIST20 (0.727/0.726); scaffold split separates them
(0.699 vs 0.682/0.669). Retrieval (50 PubChem high-Tanimoto decoys): +46% relative
Top-1 on NPLIB1 (29% vs 20%); CASMI22 Top-1 12.9% vs 8.6% next-best. ~1 CPU-s/mol.
Code+weights: `github.com/samgoldman97/ms-pred`, **MIT**, pretrained downloadable,
single-GPU <3 h/module training. Offline verdict: **best distill candidate**.
NP-strong (our test domain), adduct-conditioned, H-shift head already models what our
±2H hack approximates. Cost: 711k pool × 1 CPU-s ≈ 8 CPU-days → GPU-batched offline
only; store top-k fragment (mass, intensity) bags.

**FraGNNet (Young 2025, TMLR).** Exact method: deterministic recursive bond-breaking
on the heavy-atom skeleton to depth d (3/4) + H-count tolerance j=4 per node
(≈ our ±H idea, principled: 2j+1 formulae/node) → fragmentation DAG; two-stage GINE
(Molecule-GNN → Fragment-GNN, edge terms ablated away) predicts P(node) and
P(formula|node); spectrum = formula-mass mixture of narrow truncated Gaussians
(CFM-style mass-analyzer error model); OS (outside-support) peak mass gets its own
cross-entropy term + latent-entropy regularizers; peak→fragment annotation via
P(node|formula) Bayes flip, top-1 agreement ≥82% (91% modulo isomorphism). Training:
NIST20, InChIKey + Murcko-scaffold splits, 5 seeds. Metrics (scaffold split, hardest):
CBIN 0.678 / CHUN 0.654 vs ICEBERG 0.636 / MassFormer 0.562 / NEIMS 0.546;
retrieval Top-1 30.3% vs ICEBERG 27.9% (scaffold); annotation F1 0.89 (recall 0.81,
precision 0.98) vs GrAFF-MS 0.74 / ICEBERG 0.79. Median support only 679 formulae.
Admits blind spot: cyclizations/recombinations → OS peaks (our design §3 exploits
exactly this). Code: paper implies release via Röst group; repo not stated in text —
CONFIRM (`Roestlab` org) before use. Offline verdict: **distill candidate #2**,
best exact-mass behavior (CHUN metric); heavier enumeration than ICEBERG, GPU-offline
only.

**SCARF (Goldman, NeurIPS 2023).** Exact method: spectrum = SET of subformulae;
prefix-tree decoder emits formulae atom-type by atom-type (Thread: formulae;
Weave: formulae-differences), top-300 kept; second model assigns intensities
(cosine loss). Sidesteps bond-breaking entirely ⇒ rearrangements/ring cuts free;
100% output validity (vs ~95% binned). Training: same NIST20/NPLIB1 splits as
ICEBERG. Metrics: cosine 0.726 NIST20 / 0.536 NPLIB1; coverage 0.807/0.552;
0.21 s/mol CPU (100× faster than CFM-ID, 5× slower than binned). Code: ms-pred,
**MIT**. Offline verdict: **distill candidate #3 / fastest structured fallback**;
formulae-only output means no fragment structures for annotation, and fixed
top-300 hurts NP tails — pair with ICEBERG bags, not instead.

**GrAFF-MS (Murphy, ICML 2023).** Exact method: fixed vocabulary of ~2% of all
observed formulae (frequent product-ions + neutral losses mined from training data);
GNN → per-formula logits, incl. 3 adduct states (f, f+H2O, f+N2) and double-count
correction; linear CPU scaling vs CFM-ID quadratic; 2.8 ms/spec GPU-batched.
Training: NIST20 HCD [M+H]+/[M−H]−; test CASMI-16 (141 novel structures). Metrics:
mean cosine 0.70 vs 0.60 NEIMS vs 0.52 CFM-ID (NIST20); 0.79 on CASMI-16; P(C>0.7)
0.62 vs 0.50/0.35. "Makes human-like mistakes; distinguishes very similar compounds."
Code: no repo stated in paper; **use the MIT ms-pred FixedVocab reimplementation**.
Offline verdict: distill-via-reimplementation only; fixed vocab is a liability for
MCES-novel NPs (unseen formulae unrepresentable) — cap its GBM role to common-loss
features.

**NEIMS (Wei 2019).** Exact method: ECFP-counts(4096, r=2) → 7×2000 residual MLP;
forward (m/z-indexed) + reverse (M−m/z-indexed, i.e. neutral-loss-indexed) heads
fused by per-bin sigmoid gate; mass-weighted MSE matching the Stein cosine.
Training: 240,942 NIST-17 EI spectra. Metrics: recall@10 91.8% (NIST17, 5 Da mass
filter); beats CFM-EI 92.7 vs ~89 @10 at 0.47 ms vs 300 s/mol. Code:
`brain-research/deep-molecular-massspec` (Google Brain; license per repo, VERIFY).
Offline verdict: **not directly usable** (EI, unit-mass bins, wrong ionization for
timsTOF-ESI) — but the reverse-head idea is the intellectual parent of our scorer #1:
index predictions by neutral loss, not fragment mass, so cleavage knowledge transfers
across precursor masses.

**MassFormer (Young 2023).** Exact method: pretrained graph transformer (GROVER-style
self-supervision) + self-attention over atoms for binned spectrum regression;
attention weights double as fragment-peak explanations. Training: NIST20 [M+H]+
(70/…), MoNA + CASMI-16 generalization tests, InChIKey and scaffold splits, 10 seeds.
Metrics: best DL baseline of its generation on NIST splits (examples 0.91/0.99 cosine
on showcase molecules; systematically >CFM/FP/WLN in Fig 2c), but later beaten by
SCARF/ICEBERG/FraGNNet on identical splits (e.g. scaffold CBIN 0.562 vs 0.678
FraGNNet). Code+weights: `github.com/Roestlab/massformer`, **BSD-2-Clause**,
Zenodo data DOI 10.5281/zenodo.7874421. Offline verdict: **skip for distillation**
(superseded, [M+H]+-centric, 1 Da bins lose timsTOF-exact-mass edge); keep its lesson:
pretrained graph embeddings help most on scaffold splits (= our hidden set).

**Fragmentation-tree theory (Böcker/Dührkop; reloaded 2015).** Exact method: nodes =
molecular formulae (not structures), edges = neutral losses; MAP scoring with
learned priors (mass-error Gaussian MA≈10 ppm, noise-peak Pareto, loss-mass
distribution) + DP/ILP for best tree; used for formula ID (SIRIUS) and tree alignment
(FT-BLAST). Data: GNPS 2,006 + Agilent sets (85–980 Da, CHNOPS vs FClBrI batches).
Metrics: correct formula Top-1/Top-5 curves beat old scoring and SIRIUS2 on both
sets (Fig 4); FClBrI batch harder. Code: SIRIUS framework (academic-free, commercial
terms for SIRIUS5+ — VERIFY; not vendored here). Offline verdict: **import ideas,
not code** — loss-mass priors and tree-DP scoring shape our scorer #1; formula-ID
curves justify the formula-mass index upgrade (pool-frag-validation §4.2).

**MetFrag relaunched 2.2 (Ruttkies 2016).** Exact method: bond-dissociation
enumeration (tree depth default 2) + score(m/z, intensity, **BDE**) + 5 neutral-loss
rules for rearrangements; 2.2 adds adduct-aware PRODUCT ions ([M+Na]+/[M+K]+/
[M+NH4]+, [M+Cl]−/[M+HCOO]−/[M+Ac]−), substructure inclusion/exclusion scores,
RT/logP fusion hooks, faster search (deeper trees on ring systems, less RAM).
Data: 473 Orbitrap HR-MS/MS (359 stds) + 824-spectra verification sets. Metrics
(PubChem candidates, pessimistic ranks): Top-1 22% vs MetFrag2010 15% vs CFM-ID 9%;
+CFM-ID fusion (0.67/0.33) → 13% Top-1, mean RRP 0.901; median rank 8→4. Runtime:
75 min vs CFM-ID 12,570 min (84 queries). Code: `c-ruttkies.github.io/MetFrag`
(Java CLI + web; license per repo, VERIFY). Offline verdict: **closest to our
kernel — port, don't depend**: BDE table + adduct product-ion masses + depth switch
are pure arithmetic (scorer #1/#3); no JVM in kernel.

**MAGMa (Ridder 2014) — second-hand.** Method (via ICEBERG §Methods, FraGNNet §4.2,
MetFrag §Background): atom-removal enumeration + H-rearrangement tolerances +
substructure-restricted candidate scoring; the canonical "explain peaks with
substructures" baseline all learned methods bootstrap from. Offline verdict: ideas
already absorbed via ICEBERG's canonical-DAG recipe; no retrieval attempted
(J-STAGE/PMC endpoints failed) — do not cite numbers from it.

**FIORA (Nowatzky 2025).** Exact method: GNN edge-property prediction — SINGLE bond
break ⇒ fragment-ion/neutral-loss pair, with per-break H-rearrangement distribution
+ precursor survival + RT/CCS heads; covariates: ion mode, continuous CE (0–100 eV),
instrument; RGCN-6 (bond types matter; depth 4–6 sweet spot ≈ 6-ring coverage).
Training: NIST17 + MS-DIAL + MSnLib + CASMI16/22, curated CE metadata. Metrics
(median cosine): test+ 0.81 (vs ICEBERG 0.72, CFM-ID 0.67), MSnLib+ 0.65, CASMI16+
0.77; CASMI22+ collapses for ALL tools (0.29 — dataset-quality flag, corroborates
ICEBERG); retrieval §: non-precursor cosine discriminates best. Code+weights:
`github.com/BAMeScience/fiora`, **MIT**, MSnLib-80% weights posted. CPU note:
GPU-first design. Offline verdict: **best per-bond-probability distill target**:
single-step outputs map 1:1 onto BDE-ordered beam weights (scorer #1); both
polarities + CE covariate beat ICEBERG's positive-only/average-spectrum limits;
caveat: single-step only (no cascades) → complement, not replacement, for depth-3
enumeration.

**SingleFrag (Pérez-Ribera 2025).** Exact method: 1,000 independent binary
classifiers (ANN/GNN/combined, few-k params each) for the 1,000 most frequent
0.01-Da bins (m/z 29–269, 60% of all peaks); spectrum = presence/absence vector;
argues presence > intensity (intensities CE/instrument-noisy; cosine overweights
them). Training: 30,191 cmpds (HMDB/METLIN/MassBank/MoNA/NIST/Riken),
24.5k/3k/2.7k split. Metrics: mean per-bin F1 0.411 ANN (vs 0.290 GNN — structure
editorial: simple wins); spectrum precision > CFM-ID/3DMolMS/MassFormer at matched
recall; annotation Top-1 38%, Top-5 72% (mass-compatible candidates). Code:
`github.com/MaribelPR/SingleFrag`, **NO license file → default all-rights-reserved**;
paper CC BY-NC. Offline verdict: **reimplement idea, reuse nothing**: add a
presence-weighted (binary-match) frag column alongside sqrt-intensity (their §4
supports it); per-bin classifiers are 1k-model overhead we skip — our GBM already
fuses channels.

**MassSpecGym (Bushuiev 2024).** Benchmark, three challenges (de novo generation /
retrieval ≤256 same-mass candidates / spectrum simulation) on GNPS+MoNA+MassBank
(≈482k spectra) with MCES-disjoint splits. Numbers that matter: simulation —
FraGNNet cosine 0.52, retrieval-via-simulation Hit@1 46.6% (vs 8.4% fingerprint-FFN);
retrieval — best learned method Hit@1 ≈ 5–6.6% (DeepSets+Fourier), random 0.4–3%.
Lesson: exact-ID retrieval at scale is brutally hard (validates our retrieval+rerank
framing and small-pool doctrine), simulation-mediated retrieval beats
spectrum→fingerprint (validates the frag channel's existence), and MCES/scaffold
splits are the only honest validation (already our protocol).

## 2. Cross-cutting findings (what the 13 papers jointly say)

1. **Depth 3–4 + H-tolerance covers the observable space.** FraGNNet d∈{3,4}, j=4
   captures "most total peak intensity"; ICEBERG trains on MAGMa depth-3 DAGs;
   MetFrag default depth 2 under-covers rings (their own 2.2 fix). Our max_breaks=2
   with NO H-model beyond ±2 Da coincidence is below the literature floor.
2. **Single-bond-break signal dominates; pairs/triples are refinements.** FIORA's
   single-step SOTA (0.81 test+) and NEIMS-reverse (neutral-loss-indexed) both say:
   enumerate ALL 1-cuts first (O(n), cheap), spend beam budget on ordered 2–3-cuts.
   Our pair-first BFS has it backwards for cost/order.
3. **Bond strength must order the search.** MetFrag scores BDE explicitly; ICEBERG
   finds C–O/C–N ≫ C–C breakage empirically; CFM-ID 4.0's topology features carry
   65% of its gain. Uniform-cost BFS wastes enumeration on strong bonds and scores
   strong-bond fragments equally — the single biggest physics gap in MetFrag-lite.
4. **Rearrangements are H-shifts + a short rule list, not magic.** ICEBERG (±6H
   learned shifts), FraGNNet (j=4), MAGMa (H-tolerances), MetFrag (5 neutral-loss
   rules), GrAFF-MS (frequent-loss vocab incl. f+H2O/f+N2 adduct states) converge:
   H-shift ladder + ~10 common losses (H2O, NH3, CO, CO2, HCOOH, CH3OH, …) +
   McLafferty/rDA SMARTS cover most non-BFS peaks. FraGNNet's residual OS peaks
   (cyclizations) bound the ceiling — expect +recall, not miracles.
5. **Rings need ordered 2-cuts, not pair enumeration.** CFM-ID 4.0 sequential ring
   cleavage (+35% of gain); FIORA RGCN-6 ≈ 6-ring receptive field; our pair-BFS
   generates ring fragments only when both cut bonds are in the sampled pair, with
   no ordering prior — systematically under-scores flavonoid/glycoside-type test NPs.
6. **Adducts belong in fragments, not just precursors.** MetFrag2.2 (product-ion
   adducts), CFM-ID 4.0 (3 adduct types in rules), GrAFF-MS (adduct-state vocab)
   vs CFM-ID-2014/ICEBERG ("adducts precursor-only" assumption). Our kernel already
   neutralises via a 30-adduct table — extend the same table to fragment ionisation.
7. **Presence beats intensity for ranking; intensity for scoring.** SingleFrag
   (binary F1 0.41, Top-1 38%) + FIORA (non-precursor cosine discriminates best) +
   our sqrt-intensity hack all point the same way: score candidates with a
   presence-first term, keep intensity for tie-breaking inside the GBM.
8. **Distillation is proven; in-kernel learning is not.** Every learned method needs
   GPU at train AND ~0.2–1 s/mol at inference (SCARF 0.21, ICEBERG ~1, GrAFF-MS
   1.3 CPU-s). None fits 9 h CPU inference over 711k×56 windows. All are MIT/BSD
   with weights → precompute per-pool-structure fragment bags once on GPU, ship
   `.npy`, O(1) lookup in cell 9. GrAFF-MS's ChEMBL math (2 h/GPU vs 4 days/64-CPU
   CFM-ID) bounds our distill cost at ~1 GPU-day.

## 3. NEW scorer designs (ranked; all RDKit+numpy, no new deps)

Shared context: frag channel today = Class-2 MRR 0.259 solo, corr(analog,frag)=0.058,
14 ms/cand, `max_bonds=34` guard, ±2H shifts, MZ_TOL 0.01 Da. Predicted effects assume
the GBM keeps ~current frag-block weight and corr stays <0.15 (all three designs use
physics orthogonal to library similarity).

### Design A — BDE-ordered beam cleavage with neutral-loss ladder (BEST BET)

Replace uniform 1–2-bond BFS with: (i) per-bond BDE lookup (MetFrag-style table, kJ/mol:
C–C 346, C–N 305, C–O 358, C–S 272, C–X down to C–I 213, C=C 614, C≡C 839, arom 518,
ring bonds +strain surcharge, amide C–N 308→ cleavage-resistant flag) → breakage
priority p ∝ exp(−BDE/RT_eff), RT_eff tuned once on the NP holdout; (ii) beam search:
ALL 1-cuts (O(n), FIORA-justified) + top-K 2-cuts by summed BDE + depth-3 extension
only for candidates with <25 bonds and only along the weakest path (FraGNNet-d3
justification); ring bonds only cut as ordered pairs sharing a ring (CFM-ID-4.0
sequential rule, kills the combinatorial ring-pair explosion); (iii) mass matching
against query peaks at fragment mass AND fragment−{H2O, NH3, CO, CO2, HCOOH, CH3OH,
C6H12O6-loss for glycosides} (GrAFF-MS frequent-loss vocab, 7 entries, adduct-aware
variants in Design C); (iv) score = Σ_matched sqrt(I) · exp(−BDE_path/RT_eff) ·
loss_prior(loss) + presence_bonus·(#distinct matched peaks)/(#query peaks), with
loss_prior from tree-loss distributions (Böcker Fig-loss-mass) instead of uniform.
Predicted effect: frag solo 0.259 → **0.30–0.33** (CFM-ID-4.0 precedent: topology+ring
= +90% precision with Recall held; we port ~half that machinery) → **+0.003–0.005 LB**
via the frag block, corr(analog,frag) stays ~0.05–0.10 (BDE signal is per-candidate
physics). CPU: BDE table O(1)/bond; enumeration ≤ today's pair-BFS for ≤34-bond mols
(beam K=40 caps it), depth-3 only on small cands; est. 10–20 ms/cand, fits MPool(4).
Risk: RT_eff overfits holdout → tune on 250-NP split, freeze; BDE table ignores
charge-localisation (accepted; Design C partially covers).

### Design B — Rearrangement-augmented virtual fragments (HIGHEST ORTHOGONALITY)

Keep today's BFS masses, ADD virtual masses from ~10 SMARTS rules run per candidate:
McLafferty (γ-H to C=O → enol + alkene pair), retro-Diels–Alder (cyclohexene SMARTS
→ diene + dienophile), H2O loss per aliphatic –OH (glycoside cascades: up to 3
sequential), NH3 per primary amine, CO per non-ring ketone/aldehyde, CO2 per
carboxylic acid, CH3OH per methyl ester, plus ±1..4H ladder on every BFS fragment
(FraGNNet j=4 replaces today's ±2 coincidence window). Each virtual mass carries
rule_prior (McLafferty 0.9, rDA 0.8, cascades 0.7^k, H-shifts 0.6^|δ|) multiplying
sqrt(I) on match; separate presence-count feature (SingleFrag lesson) so the GBM can
learn "rearrangement-explained peaks count double". Predicted effect: frag solo →
**0.29–0.32**, but concentrated on the current frag-blind tail (rearrangement-driven
spectra where analog also struggles) → **+0.002–0.004 LB** with near-zero corr to
both analog AND Design A (different peak subsets explained). CPU: SMARTS matching is
RDKit-C++ fast (~µs–ms/mol); +~2 ms/cand. Risk: rule false positives on dense windows
(mitigate: rule_prior calibration + cap virtual masses at 40/cand); cyclization OS
peaks (FraGNNet's admitted blind spot) remain uncovered — state as ceiling.

### Design C — Adduct-switched charge-localized ionisation (CHEAPEST, DO FIRST)

Extend the existing 30-entry adduct table from precursor neutralisation to FRAGMENT
ionisation: for the query's adduct, generate fragment masses under charge-retention
models — proton sticks to most-basic site (heteroatom count/basicity proxy: N > O >
S; MetFrag2.2 precedent), Na+/K+ retained on O-rich fragments (O-count ≥ loss O-count
else fragment dropped for Na-adduct queries), negative mode mirrors with
Cl−/formate/acetate adduction on acidic fragments; neutral-loss ladder from Design A
conditioned on adduct (Na+ spectra: neutral losses identical, charge tag differs —
megayak's adduct-aware fix, formalised). Score term: adduct_consistency = fraction of
explained intensity carrying the query's charge model. Predicted effect: frag solo →
**0.27–0.30** alone, but its real value is interaction: fixes the adduct-mislabel
tail that currently triggers the 30 ppm fallback AND upgrades Design A matches →
combined A+C **+0.004–0.006 LB**. CPU: pure arithmetic on existing BFS masses,
<1 ms/cand. Risk: basicity proxy is crude (no pKa calc in kernel) — gate by polarity
+ adduct class, keep weights small.

Ranking: **C first (1 day, unblocks A), then A (the MRR engine), then B (tail
coverage); A+C+B stacked est. frag 0.259 → 0.32–0.36, ≈ +0.006–0.010 LB** if corrs
hold. All three preserve the 0.058-corr complementarity thesis: they add physics the
library channels cannot see. Separately, offline-distill FIORA (MIT, both polarities,
per-bond probs → Design A beam weights) + ICEBERG (MIT, NP-strong bags) on one
GPU-day; ship two `.npy` lookups as Design D (est. additional +0.003–0.006 LB per
pool-frag-validation §4.7).

## 4. What NOT to port (explicit)

- In-kernel neural inference (any GNN/transformer): 0.2–1 CPU-s/mol × 56 cands ×
  thousands of queries ≫ 9 h. Distill, don't infer.
- Full CFM-ID EM retraining or MAGMa exhaustive enumeration: months of compute for
  weights we can download (MIT) instead.
- Fixed-vocabulary-only scoring for NPs (GrAFF-MS): unseen-formula blindness is
  exactly our MCES-novel failure mode; use as features, never as the support.
- EI/binned/1-Da machinery (NEIMS/MassFormer bins): timsTOF exact mass is our edge;
  all new masses computed at full precision, matched at 0.01 Da as today.
- Precursor-intensity fitting (FIORA §retrieval: precursor hurts discrimination):
  keep precursor mass as anchor, down-weight its intensity in frag features.
- Commit nothing: all designs are kernel-edit proposals with frozen-before-submit
  constants, per validation discipline (seed-bagged GBM, ≥0.007 significance bar).

*Do NOT commit — analysis only.*
