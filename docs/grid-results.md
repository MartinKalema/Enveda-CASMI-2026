# Grid results: upgrade-combo search (v-grid, branch feat/exp-grid)

Plain English: we tested every upgrade axis one variable at a time against a
pinned base, then ranked the combinations. Validation is grouped by molecule
(no structure shared between database and queries), 110 queries on each of 2
seeds, with the plain cosine baseline scored on every split. A change only
ships if it beats the baseline by at least 0.012 on BOTH seeds.

## What the validation regime means (read before the numbers)

- This grid measures the **novel-molecule regime**: every query structure is
  held out of the database, so library channels (cosine, entropy) score ~0 on
  the truth by construction. Absolute MRRs here are far below LB numbers
  (which mix banked and novel spectra). Only the gaps vs the anchor transfer.
- Queries are train structures, which flatters train-derived signals
  (fingerprint model may have seen them; formula priors are deflated for
  held-out rows, so the prior feature was dropped). Treat fp/GBM magnitudes
  as optimistic; their rank order across cells is what counts.
- Rank-fusion cells (non-GBM) rerank the top-100 of the floor base. Tight
  windows pay their own recall cost (no truth force-keep); truth pass rate at
  8.5/10/20ppm is 0.98/0.99, so the window comparison is nearly recall-free.

## Ranked combo table (mean MRR@25; hit@100 and median rank as backup)

| combo | s10 | s11 | d10 | d11 | verdict |
|---|---|---|---|---|---|
| GBM-F25 | 0.6745 | 0.7278 | +0.6745 | +0.7262 | SHIP over F8 |
| GBM-F31 | 0.6993 | 0.7003 | +0.6993 | +0.6987 | hold (split vs F25: +0.025/-0.028) |
| GBM-F8 | 0.6046 | 0.6346 | +0.6046 | +0.6330 | SHIP over F3 |
| GBM-F3 (cos,ent,ana) | 0.1193 | 0.1718 | +0.1193 | +0.1702 | SHIP over anchor |
| all-on, ent floor | 0.0511 | 0.0855 | +0.0511 | +0.0839 | SHIP |
| window 8.5ppm | 0.0386 | 0.0855 | +0.0386 | +0.0839 | SHIP vs 20ppm, tie vs 10ppm |
| window 10ppm | 0.0305 | 0.0832 | +0.0305 | +0.0816 | SHIP vs 20ppm |
| window 20ppm | 0.0186 | 0.0553 | +0.0186 | +0.0537 | SHIP vs anchor only |
| ent floor + analog | 0.0223 | 0.0490 | +0.0223 | +0.0474 | SHIP |
| ent floor only | 0.0110 | 0.0282 | +0.0110 | +0.0266 | hold (s10 just under) |
| fp-cos fill | 0.0026 | 0.0264 | +0.0026 | +0.0248 | hold (fusion; ships only inside GBM) |
| all-on, cos floor | 0.0004 | 0.0153 | +0.0004 | +0.0137 | hold |
| dedup-ik14 | 0.0019 | 0.0074 | +0.0019 | +0.0058 | hold (direction +, under bar) |
| analog K/p (all 6) | 0.0000 | 0.0066-0.0069 | +0.0000 | +0.0050-0.0053 | hold (axis flat) |
| cap 2000 vs none | identical | identical | — | — | hold (no-op at these windows) |
| metfrag any tol | 0.0000 | 0.0046 | +0.0000 | +0.0030 | hold (flat across 0.005/0.01/0.02; fusion-negative) |
| anchor cos-only | 0.0000 | 0.0016 | — | — | anchor |

## Axis verdicts

- **Analog K/p**: flat. K=50/100/200, p=3/4 all within 0.0005. Keep K=100/p=3.
- **Window**: 8.5 ~= 10 > 20 on both seeds, recall cost ~1-2%. Keep **10ppm**
  (matches the fork; 8.5's edge over 10 is under the bar, and community ppm
  audits warn some spectra are mis-indexed >10ppm, so 10 is the safer pick).
- **Cap**: 2000 vs none identical (windows rarely exceed a few hundred
  candidates). No cap needed.
- **MetFrag**: tol axis flat (0.005=0.01=0.02); as a fusion fill it scores
  below base (noisy-channel dilution). Keep 0.01 as a **GBM-only feature**.
- **FP fills**: fusion hold; fp-cosine is the strongest single fill inside the
  GBM (per-feature AUC 0.89). Ship as ranker feature, not as fusion weight.
- **GBM sets**: F8 >> F3 on both seeds (fp + mass-error + analog-best carry
  it); F25 > F8 on both seeds (+0.07/+0.09); F31 split vs F25. Ship **F25**.
- **Dedup**: inchikey14-grouping leans positive but under the bar. Keep
  exact-smiles.
- **Floor**: entropy floor + analog ships; cosine floor earns ~nothing here
  (expected: disjoint regime). Mixed floor does not ship. Entropy floor is
  recommended **as an experiment**, flagged: LB mixes banked spectra where the
  cosine floor earns, so this needs an LB check before it touches v14 proper.

## Recommended v14 config (only shipped changes)

1. Floor: entropy top-100 base (experimental flag: confirm on LB).
2. Window 10ppm, no candidate cap.
3. Analog K=100/p=3 (unchanged).
4. Fills via GBM-F25 ranker (cos, ent, 6 analog variants, 3 metfrag tols,
   fp dot/top/cos, mass error, analog-best/cos gaps, top-1 gaps, mass/prior
   shape features). No formula-prior feature (holdout artifact).
5. MetFrag tol 0.01, fp-cosine: ranker features only.
6. Dedup exact-smiles (unchanged).

## Leaks caught and fixed during the grid (so nobody re-learns them)

1. Candidate masses from full-train medians included the query's own held-out
   spectra (mass-error AUC 0.997). Fixed: DB-only medians + formula-mass
   fallback for novel structures (the v7 formula index, which is the point).
2. Missing-mass fallback of 0.0 made truth the max-error candidate
   (AUC 0.000 the other way). Fixed by the formula fallback.
3. Formula-prior feature was a holdout artifact (held-out rows deflated).
   Dropped from all sets.
4. Tight-window cells first looked free because truth was force-kept in the
   pool. Removed: windows now pay recall.

## Reproduce

- `.venv/bin/python v-grid/run_grid.py cache` (~15 min, writes
  `v-grid/cache/seed{10,11}.npz`, one row per query per channel, no recompute
  after).
- `.venv/bin/python v-grid/run_eval.py` (fusion grid + GBM leave-one-seed-out
  + `v-grid/grid_table.txt`).
- Raw numbers: `v-grid/grid_table.txt`. Method: `v-grid/README.md`.
