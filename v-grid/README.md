# v-grid: staged combo grid with cached channels

## Method (one variable per experiment, anchor-gated)

1. `run_grid.py cache`: grouped-by-molecule splits (inchikey14; 110 queries x
   seeds 10, 11). Widest candidate pool cached once per query (20ppm,
   expanded to >=150, hard cap 2000, mass-sorted): cos, ent, analog x6
   (K=50/100/200 x p=3/4 via one packed-bit Tanimoto matrix), metfrag x3
   (tol 0.005/0.01/0.02, vectorized searchsorted), fp dot/top512/cos from
   `data/fp_trans.pt`, formula-index masses (DB-only medians, never full-train),
   truth + floor meta. `v-grid/cache/seed{10,11}.npz`.
2. `run_eval.py`: fuse cached columns without recompute (rank-average rerank
   of floor top-100; GBM scores rank directly). v0-cosine anchor on every
   split. Ship iff MRR@25 delta >= +0.012 on BOTH seeds. hit@100 + median rank
   as backup resolution. GBM sets {3,8,25,31} trained leave-one-seed-out
   (positives + 60 neg/query, HistGB-200).
3. Output: `v-grid/grid_table.txt`, verdicts in `docs/grid-results.md`.

## What not to re-learn

- Candidate masses must come from DB-only medians + formula fallback; the
  0.0 fallback and full-train medians are both perfect-separator leaks.
- Formula-prior is a holdout artifact (held-out rows deflated). Dropped.
- Window/cap cells must not force-keep truth; they pay recall.
- Disjoint regime: library channels score ~0 on truth by construction.
  Absolute MRRs are pessimistic vs LB; only anchor gaps transfer.
