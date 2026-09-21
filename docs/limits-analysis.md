# First-principles limits analysis: where the points are and what caps us
Date: 2026-09-21 | Project: enveda-casmi26-molecule-id
All numbers below were computed from local data (`data/train.parquet` 2.54M rows,
`data/test.parquet` 1213 rows / 400 molecules, `data/coconut_cands.parquet` 695K rows)
with the project `.venv`, read-only. Assumed adduct offsets (Da):
[M+H]+ 1.007276, [M+Na]+ 22.989218, [M+K]+ 38.963158, [M+NH4]+ 18.033823,
[M-H]- -1.007276, [M+Cl]- 34.969402, [M+CH2O2-H]- (formate) 44.998204.

## 1. MRR@25 math: what rank distributions give what scores

MRR = average over 400 molecules of 1/rank of the FIRST correct guess (0 if none
of the 25 hit). Rank values: 1st = 1.0, 2nd = 0.5, 3rd = 0.33, 4th = 0.25,
5th = 0.20, 10th = 0.10, 25th = 0.04. Only the first correct guess counts;
wrong guesses cost nothing except the slot they occupy.

What this means per molecule (each molecule is worth 1/400 = 0.0025 of MRR):
- Moving one molecule from a miss to rank 1 gains 0.0025. Miss to rank 25 gains
  only 0.0001 (25x less). Rank 2 to rank 1 gains 0.00125.
- Current leaderboard gaps are ~0.02 (top 0.409 vs 0.388). That is just EIGHT
  molecules moved from miss to rank 1, or sixteen moved from rank 2 to rank 1.
  Small rank improvements across many molecules beat heroic rescues of a few.
- Our v0 scores 0.088. That equals roughly 35 molecules hitting at rank 1 (or,
  equivalently, ~70 at rank 2, or ~140 molecules each contributing ~0.25).
  Concretely: v0 already gets some molecules right, so some test answers (or
  near-exact analogs) ARE reachable from train candidates. Train recall is not zero.
- Scoreboard arithmetic for planning: 100 molecules at rank 1 + 100 at rank 4 +
  200 misses = (100 + 25 + 0)/400 = 0.31. To beat 0.409 we need roughly
  160+ rank-1 equivalents out of 400. Everything past rank ~5 (0.20) is nearly
  worthless per slot, so optimisation priority is: (a) get the truth into the 25
  (recall), then (b) push it to rank 1-3 (precision). Recall without ranking pays
  almost nothing: 400/400 molecules hitting at rank 25 still only scores 0.04.

## 2. Candidate recall ceiling: mass-window counts per test molecule

Neutral masses from precursor minus adduct agree across each molecule's spectra
to better than 6 ppm (median spread 0 ppm; zero molecules above 20 ppm), even
for the 79 molecules seen in both positive and negative mode. Mass inference is
free and reliable: the funnel starts from a trustworthy neutral mass every time.
Test neutrals span 246-439 Da, median 328 Da.

Candidate counts per molecule (median / mean / p10 / p90 / max over 400 mols):
- Train 277K structures, +/-10 ppm: median 78, mean 112, p10 18, p90 257, max 484.
- Train 277K structures, +/-20 ppm: median 126, mean 158, p10 30, p90 340,
  max 506, min 2.
- COCONUT-new only (627K structures), +/-20 ppm: median 55 extra per molecule.
- Train + COCONUT-new combined, +/-20 ppm: median 188, mean 234, p90 466,
  max 939.
- Molecules with fewer than 25 candidates (slots cover the whole pool):
  7.0% train-only, 3.3% combined. Molecules with more than 2000: ZERO at either
  tolerance. Molecules with fewer than 10 combined: 0.5% (2 molecules).

Three conclusions. First, pool SIZE is not the problem: nobody faces millions
of same-mass candidates; the median pool is 188 and the worst is 939, both
rankable inside 9 hours with precomputed embeddings. Second, COCONUT helps
recall but costs ranking: it grows the median pool ~50% (126 -> 188) and only
271K of its 627K new structures (43%) even fall in the 246-439 Da test range
(the DB median mass is 431 vs test median 328; train median 333 matches test
far better). If the truth is a known natural product missing from train,
COCONUT is the only place it can come from; if the truth is truly novel
("dark matter"), neither pool contains it and no ranker can score. Third, the
formula bottleneck is extreme: the median 126-structure pool collapses to a
median of just 9 distinct molecular formulae (99.5% of molecules have <25
formulae in-window; median ~16 isomers per formula). A perfect formula predictor
shrinks every pool ~14x, but formula alone can never rank within the ~16
isomers left, so formula -> structure-rerank is the mandatory two-stage shape.

Honest bound on recall: v0 (train-only, cosine) already scores 0.088, proving
part of the test set is reachable from train. The rest is unknowable without
labels, but the chemistry says train is 99% out-of-domain (Section 4). Expect
train-only recall ceiling well under 50%; COCONUT adds the only known-NP
recall lever (+55 median candidates); true dark matter needs generative guesses.

## 3. Slot strategy under 25 guesses: diversity vs precision

Median pool 188 vs 25 slots means ranking decides ~97% of molecules; only 3.3%
can enumerate their whole pool. Slot math per molecule:
- Slots 1-3 carry ~70% of the maximum points a molecule can give
  (1.0 + 0.5 + 0.33 = 1.83 of 2.63 total over all 25 ranks... and ranks 6-25
  combined are worth less than rank 2 alone: 0.5 vs 0.48). Front-load the best
  calibrated guess first; never sacrifice rank-1 confidence to "cover" more.
- Only the FIRST correct guess pays, so near-duplicate structures in nearby
  slots are pure waste: two isomers of the same skeleton in slots 3 and 4 earn
  nothing extra over one of them. Fill slots with structurally DIFFERENT skeletons
  (distinct InChIKey14). Train SMILES are already ~99.4% stereo-collapsed
  (277,547 SMILES -> 275,791 Key14s; window inflation only ~1.00x), so dedup is
  cheap hygiene, not a big points lever, but still mandatory.
- Generative (de novo) guesses: literature says top-1 exact generation accuracy
  is 0.00 for all baselines, and our pools already contain 188 median real
  candidates. A generative guess in slots 1-15 displaces a real candidate with
  far higher hit probability and negative expected value. Use generative fills
  ONLY in trailing slots (roughly 16-25, worth 0.06 down to 0.04 each) and ONLY
  for molecules where retrieval confidence is flat (no candidate scores above
  threshold), i.e. suspected dark matter. Confident molecules: all 25 slots
  retrieval, diversified. Unconfident molecules: retrieval ranks 1-15, diverse
  analogs/generative 16-25 as lottery tickets.
- Special case: the ~3% of molecules with <25 combined candidates. Submit the
  whole pool (deduped) and fill remaining slots with mass-neighbour analogs from
  outside the window, most similar first. Leaving slots empty can never help
  (no penalty for wrong guesses).

## 4. Instrument and adduct hard constraints (what caps accuracy)

Instrument: test is 100% Bruker timsTOF. Train has 1.15M timsTOF spectra, which
looks like coverage but is a mirage: essentially all of it is enveda-180
(synthetic drug-like chemistry, wrong space) plus 1,184 enveda-np-examples
spectra (right instrument AND right chemistry, but only 250 structures). The
chemistry-closest large libraries (riken plant metabolites 347K, gnps 221K) are
Orbitrap/QTOF community data. Any model trained naively on all 2.5M spectra
learns synthetic-chemistry fragmentation on timsTOF plus NP fragmentation on
other instruments, and is asked at test time for NP fragmentation on timsTOF, a
combination barely present in training. This domain gap, not model size, is the
accuracy cap. Practical consequences: condition every model on instrument +
collision energy, upweight timsTOF + riken/gnps/np-examples, and validate on
structure-disjoint splits (stereo-independent Key14 at minimum) or local scores
will lie (v1: local MRR 0.099, real LB 0.044).

Adducts (spectra: 79% [M+H]+; molecules: 258/400 [M+H]+-only, 79 dual +/-,
25 [M-H]--only; rare: 5 [M+Na]+-only, 2 [M+NH4]+-only, 2 formate-only,
1 mixed [M+K]+, 1 mixed [M+Cl]-):
- [M+H]+ (258 molecules, 65%): 245K train structures, 589K timsTOF spectra.
  No data cap; this is where most points live. Win or lose here.
- [M-H]- (in 107 molecules incl. 25-only): 99K structures. Adequate.
- [M+Na]+ (16 molecules touch it, 5 only): 25K structures / 16K timsTOF spectra.
  Thin; expect degraded ranking, keep mass tolerance tight (Na+ mass error
  behaves differently) and lean on the molecule's [M+H]+ spectra when present
  (9 of 16 have them).
- [M+NH4]+ (3 molecules: 2 only, 1 mixed): 9K structures / 4.8K timsTOF spectra.
- [M+K]+ (1 molecule, mixed with [M+H]+): 4.5K structures / 1.1K timsTOF spectra;
  gold np-examples has just 16 [M+K]+ spectra total. Data-driven K+ scoring is
  unlearnable; backstop with rules (mass + neutral losses) and let the
  molecule's [M+H]+ spectra carry the ranking.
- [M+Cl]- (1 molecule, mixed): 7K structures / 2.4K timsTOF spectra; np-examples
  has 7. Same verdict as K+.
- Formate [M+CH2O2-H]- (17 molecules touch it): 21K structures. Workable.
Bottom line: ~8 molecules (2%) touch K+/Cl-/NH4+-only territory where learned
scoring has almost no support; ~30 dual-adduct molecules get a free second view
of the same neutral mass (masses agree <6 ppm), which should be fused, not
averaged away. Collision energy is NOT a constraint (all four test CE patterns
exist abundantly in train). Spectrum density IS a shift to handle: test spectra
carry median 230 peaks vs train median 42 (timsTOF train median 138); raw
cosine-style scorers tuned on sparse spectra will underperform on dense test
spectra without renormalisation or top-N peak truncation.
