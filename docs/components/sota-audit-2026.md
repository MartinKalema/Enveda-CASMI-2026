# SOTA audit 2026: are our per-area picks still current?

Date: 2026-09-30. Scope: fresh 2025–2026 literature vs the incumbents in
`structure-sota.md`, `similarity-sota.md`, `analogs-sota.md`,
`fragmentation-sota.md`, `ranking-sota.md`, `denovo-sota.md`.
Corpus: `docs/papers/2026/` — 5 PDFs, all `file`-verified, with
`pdftotext -layout` extracts (`.txt` next to each PDF). Nothing committed.

| File | Paper | Verdict basis |
|---|---|---|
| `msflow_2602.19912v1.pdf` (13 pp, 1.5 MB) | Mqawass et al., MSFlow, arXiv Feb 2026 — **WITHDRAWN 12 Mar 2026 (data leakage)** | full read (abstract, Table 1, §eval) + withdrawal notice on abs page |
| `frigid_2604.16648.pdf` | Bohde et al., FRIGID, Coley group, arXiv Apr 2026 | full read (Tables 1, 2, 8, App A.3–A.4) |
| `msgym-wild_2606.19624.pdf` (8 pp) | Liu/Bushuiev et al. (MIT/Colely/Enveda/Pluskal), MassSpecGym in the Wild, arXiv Jun 2026 | full read (abstract, §3–7, Tables 4–5, recs) |
| `specreboot_2026.02.03.703446.pdf` (29 pp) | Charria-Girón et al., SpecReBoot, bioRxiv Feb 2026 v2 | full read (abstract, Figs 1–2, method) |
| `specreboot-libmatch_2026.07.30.741704.pdf` (37 pp) | Charria-Girón et al., SpecReBoot library matching, bioRxiv Jul 2026 | full read (abstract, method, benchmarks) |

Cited-but-not-downloaded (web-verified, hedged): MSFlow abs page (withdrawal
notice); FRIGID repo `coleygroup/FRIGID` (GitHub API: **no license declared**);
SpecReBoot repo `ECharria/SpecReBoot` (GitHub API: **GPL-3.0**); Khoo &
Barzilay, Nat Metab Jun 2026 (MIT "why ML fails", via T&F news summary +
PubMed record 42277271); MS2DeepScore 2.0, Nat Commun Jan 2026
(s41467-026-69083-y, published version of the bioRxiv we already read);
NMR-Solver, Nat Commun 2026 (s41467-026-71315-0, NMR-only); FlowMS, arXiv
2603.18397 (NPLIB1-only 9.15% top-1, no code statement found in text);
DiffSpectra, arXiv 2507.06853 (multi-modal IR/Raman/UV, not MS/MS-only);
Turkina et al. 20-metric study, Anal Chem 2026 (published; still abstract-only
for us); IBM BART-TTT (published, 3.16% MSGym — already P10 in denovo-sota);
ICEBERG 2.1 (Jul 2026, GPU speedup + NIST23 weights, via ms-pred README).

## Lead-by-lead verification (do not trust snippets)

1. **MSFlow 45% — DEAD. Withdrawn for data leakage.** v1 Table 1 claims
   NPLIB1 44.70% / MassSpecGym 32.00% top-1 (13.9× DiffMS). The tell is
   inside v1 itself: a plain "CDDD decoder baseline" scores 24.10% MSGym
   top-1 — impossible under leakage control — and the decoder pretraining
   corpus admits NPLIB1/MassSpecGym molecules. Authors withdrew 12 Mar 2026
   ("potential data leakage issue"). Textbook confirmation of our
   discount-everything discipline. No build change; keep as cautionary cite.
2. **SpecReBoot — REAL, but a confidence wrapper, not a kernel.**
   Nonparametric bootstrap over fragment m/z bins (B replicates, ~63% bins
   kept each) → edge/match support values. Networking paper: prunes
   unreliable edges, recovers robust low-similarity links. Library-matching
   follow-up: match-support promotes true matches +4 ranks on average vs
   cosine ranking; flags wrong annotations at AUROC 0.75 where cosine fails;
   matchms-compatible (cosine/modified-cosine/Spec2Vec/MS2DeepScore/
   FlashSimilarity). Repo is **GPL-3.0** → do not vendor; reimplementing
   bootstrap resampling is trivial statistics anyway. Build change: port the
   *idea* as CPU ranker features (per-match support, rank-stability), not
   the code.
3. **IBM test-time-tuned LMs (3.16% MSGym / 12.88% NPLIB1) — CONFIRMED,
   already incorporated** (denovo-sota P10: BART-TTT). Published (IBM
   Research page + OpenReview). Verdict stands: no TTT in kernel
   (per-query gradients too costly); plain fine-tuned checkpoint stays a
   rank-#3 fill alternative pending weights/license. No change.
4. **MS2DeepScore 2.0 cross-ion — CONFIRMED, now peer-reviewed** (Nat
   Commun Jan 2026). Same numbers/behavior as the bioRxiv we fully read;
   models + case-study data on Zenodo/WUR. Incumbent analog pre-ranker
   assessment stands. No change.
5. **MIT "why ML fails" (Khoo & Barzilay, Nat Metab Jun 2026) — REAL,
   peer-reviewed comment.** Evaluated MIST, DreaMS, NN-retrieval under
   random vs scaffold splits; three failures via data attribution:
   (a) spectra under different experimental conditions, (b) same-m/z /
   different-intensity profiles, (c) peaks from unseen fragments. Rx:
   condition-consistent datasets first, alternatives to fingerprint
   formulations. Net effect: validates our retrieval+rerank backbone over
   pure fingerprint decoding AND our max-over-replicates / adduct-gate
   handling of (a). No change.
6. **NMR-Solver / AI-for-spectroscopy — REAL but OUT OF SCOPE.**
   NMR-Solver (Nat Commun 2026): 1H/13C NMR only, no MS/MS. The "ACM 2026"
   item is the IJCAI-2025 spectroscopy survey (MS/NMR/IR/Raman/UV-Vis) —
   taxonomy, no method. Neither touches any MS/MS build decision.

## Per-area verdicts

### Similarity (library search) — CURRENT
No 2025–2026 paper displaces entropy + Flash-hybrid + dual-channel NL
(Design A) / PMI reweighting (Design B). SpecReBoot adds no new kernel —
only a bootstrap confidence wrapper around existing scores (cosine,
modified cosine, Spec2Vec, MS2DeepScore, Flash). Turkina 20-metric now
published; still supports local-network eval, no kernel implication.
**Addition (features only):** per-match bootstrap support + rank-stability
as ranker inputs (reimplement, ~B×100 similarity calls offline; GPL code
not vendored).

### Analogs / neighborhood — CURRENT (+1 feature family)
SpecReBoot-network's "remove unreliable edges / recover robust low-sim
links" is independent evidence for our D2 (shift-consensus) + D3
(calibrated reliability) direction and MS2Query's kNN-mean lesson — but
contributes no scoring math we lack. **Addition:** bootstrap edge-support
as an analog propagation weight alongside D3's logistic gate.

### Structure prediction (de novo fills, slots 16–25) — SUPERSEDED (source), build change CONDITIONAL
FRIGID is the new best *honest* de novo method, including under predicted
formula — the setting that matters for us (Table 8, MIST-CF top-5):
MSGym top-1 **14.28%** (full) / 11.60% (base) vs MIST+MolForge-corrected
**10.73%**; known-formula 18.29% vs 10.73%; NPLIB1 25.03% vs 8.34%
(DiffMS). It also independently reproduces the corrected 10.73% (App A.3),
retiring the v4-inflation dispute, and its ICEBERG-guided refinement
(DiffMS 8.34→13.33% NPLIB1) further validates ICEBERG-distillation.
Caveats blocking an immediate R1 swap: (i) **no license on
`coleygroup/FRIGID`** (all-rights-reserved by default — same severability
problem as MolForge NC, worse); (ii) 247M params, 6.58 s/spectrum on an
Nvidia 6000 Ada (GPU) — CPU cost uncalibrated, needs the SpecTUS-style
Xeon check before kernel budgeting; (iii) headline numbers are
known-formula (Table 1), predicted-formula costs ~4 pts on MSGym.
FlowMS (9.15% NPLIB1-only, no code statement) and DiffSpectra
(multi-modal, non-MS/MS benchmark) are non-contenders for our slots.
MS-GPT stays discounted (known-formula protocol); GLMR stays flagged
(the Wild audit documents exactly that artifact class).
**Build decision:** queue FRIGID as R1-swap candidate behind two gates —
license/weights clarification + CPU calibration. MIST+MolForge stays R1
until both clear. No change to slots-16–25 discipline or R2 hard filters.

### Fragmentation / forward simulation — CURRENT (strengthened)
No new forward model beats the ICEBERG (distill #1) / FraGNNet (#2) /
FIORA (per-bond) trio. Two reinforcements: ICEBERG 2.1 (GPU speedup +
NIST23 weights) cheapens distillation; FRIGID's ICEBERG-guided refinement
gain proves ICEBERG scores carry rerank signal even for generated
molecules. Wild audit adds no forward-model findings. No change.

### Ranking / fusion — CURRENT (+protocol upgrade)
No new LTR result displaces YetiLoss-MRR + QueryRMSE + RRF-Platt stack.
The Wild audit is directly load-bearing for validation hygiene: 17/26
MassSpecGym papers had eval issues (leakage / shortcut / metric bugs);
it ships **MassSpecGym v1.5** (`pluskal-lab/MassSpecGym`) with data-safe
MIST-CF, uniform RDKit canonicalization, pinned InChIKey-14 metrics with
bootstrap CIs. **Adopt v1.5 + CIs for all local fill/ranker validation**
(this is also the instrument that would have caught MSFlow and the MIST
batch-mask bug). SpecReBoot match-support becomes a candidate gate
feature (Venn-width backoff family, Design C).

## Cross-cutting lesson (new)
The 2026 literature split cleanly in two: withdrawn/inflated headline
numbers (MSFlow 45%, MIST+MolForge v4 31%, GLMR 64%) vs audited honest
numbers (FRIGID ≤18.3%, DiffMS 2.3%, MIST+MolForge-corr 10.7%). The Wild
audit (MIT/Coley/Enveda/Pluskal, MassSpecGym v1.5) is now the canonical
backstop for our "discount by formula protocol, require spectrum-blind
baseline, re-eval locally" rules. Our absolute-LB currency
(disjoint@5, |Δ|≥0.012) is unaffected.

*Do NOT commit — analysis only.*
