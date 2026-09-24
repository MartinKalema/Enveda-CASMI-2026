# Task-specific finds (0.342 tutorial + pitfalls + host baseline)

Source: alexchilton tutorial (0.342 scorer), kitopl CV pitfalls, denpugovkin
host-GPU baseline, franciscoangulo MSBuddy+MIST versions. Pulled 2026-09-24.

## New framework: Class 1/2/3 (we only had 1/2)
- Class 1: own spectra in library -> nearly always gettable.
- Class 2: structure in pool, no spectra -> rank it.
- Class 3: structure in NO database -> score exactly 0 unless generative.
- Classes SHIFT when the pool grows. Our COCONUT work moved molecules 3->2.
- Class 3 cannot be measured locally (219/220 NPs already in COCONUT).

## Tail slots 13-25 worth ~0.0005 total
Truncation study: dropping ranks 13-25 costs 0.0005 MRR. Our fill obsession
(v3-v11) fought over crumbs. Top-12 is the game; fills are lottery tickets.

## 96.9% of above-truth candidates are ISOMERS
Same formula => formula features are chance (AUC ~0.50, measured on five).
Only FRAGMENTATION separates isomers. Our MetFrag 3-seed win is the right
weapon - promote it from fills to top-12 contention.

## Offline != board (1 success in 6)
Fingerprint accuracy does not predict ranking. Validation windows must match
test sizes (theirs: holdout 82.8 vs test 131.6 cands/query). Our validations
must report mean window size alongside MRR.

## Open threads
- BRICS de novo for class 3 (tutorial cell 22, unread in full).
- MSBuddy formula versions (franciscoangulo) - formula tool trial.
- Host GPU baseline 0.174 (denpugovkin) - what the hosts think is easy.
- Pitfalls notebook: test.parquet = enveda-180 slice; inchikey14 != scored
  key; enveda-np-examples holdout leaks; disagreement-about-identity costs.
