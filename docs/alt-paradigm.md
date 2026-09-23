# Alt paradigm: formula-first isomer retrieval with no cosine floor

Branch: `feat/exp-alt` | Date: 2026-09-23

## Why this exists
Every production variant (v0-v9) contains a cosine similarity floor: rank-1..5
slots are always spectrum<->spectrum cosine matches against train spectra. That
floor is the only proven hidden-test signal (LB 0.088-0.115), but it caps
novel-scaffold retrieval: a truly new natural product has no near-duplicate
spectrum in train, so cosine can only return analogs, never the exact structure.
This experiment asks: can a pipeline with NO spectrum<->spectrum comparison at
all retrieve exact structures? If yes, its errors are uncorrelated with cosine
and it earns rerank/ensemble slots; if no, we keep the floor and stop trying
to replace it (lessons 26, 28).

## Design (MIST-CF style, survey rank 2)
Stage 1 — formula ranking: group mass-window candidates by molecular formula;
score each formula by subformula explained-intensity of the query spectrum
(top-20 peaks vs enumerated subformulae, RDBE-filtered, 15 ppm). Pure
spectrum-vs-formula physics. Keep top-2 formulae (funnel).
Stage 2 — isomer ranking: rank same-formula isomers by forward fragmentation
fit (RDKit 1-2 bond cleavage fragment masses vs query peaks, sqrt-intensity
fraction) with a neutral-loss bonus (query peaks vs 24 common-loss constants)
as tiebreak. Pure structure->spectrum physics.
Scored variants: frag-only, formula-explained-only, 50/50 blend, full
(0.5 frag + 0.3 formula + 0.2 loss). Final ranking = variant score inside the
funnel pool; funneled-out candidates rank below the pool (honest funnel:
formula errors cascade into misses, exactly the failure mode to measure).

What was deliberately excluded: cosine, entropy similarity, Tanimoto,
fingerprints, library priors, GBM rerankers trained on spectral-similarity
features. The comparison anchor (v0-style cosine) is computed on the same
split but never feeds the alt ranking.

## Validation protocol (per lessons 21-23: anchor-gated, multi-seed)
Structure-disjoint on inchikey14 groups (held groups' spectra never scored;
alt needs no library spectra by construction so leakage is impossible).
40 queries x 3 seeds (n=120). Candidates: mass window (min_n=100, cap=400)
over remaining structures + truth injected mass/formula-only. Metric MRR@25.

## Results
| variant | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| frag-only | | | | |
| formula-only | | | | |
| blend 50/50 | | | | |
| full | | | | |
| cosine anchor | | | | |
Funnel recall (truth formula in top-2): seed 0/1/2 = / / .

## Verdict
TBD after run.

## Cost / offline notes
RDKit needed for fragment enumeration (precompute once, `frag_disk.pkl` cache;
inference is numpy matching, same pattern as v7). No downloads, no weights,
no license server — fully offline-legal. MS2DeepScore/Spec2Vec (survey rank 5)
was considered and rejected for this branch: it reintroduces spectrum-embedding
cosine (a cosine floor by another name) and needs pretrained weight downloads;
formula-first is the structurally cleanest break.
