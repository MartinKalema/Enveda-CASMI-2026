# v0: mass filter + spectral cosine (LB 0.088)

Plain English: for each mystery molecule, weigh it from its spectra, pull every
known structure with nearly the same weight, then rank them by how similar
their lab spectra look to the mystery spectra. Best match first, top 25 out.

Steps:
1. Neutral mass: precursor m/z minus adduct weight (7 adduct types + dimers).
   Median over the molecule's 1-9 spectra.
2. Candidates: all 277K train structures within 20ppm (widen until 200+).
3. Score: greedy cosine between fragment peaks (0.02 Da tolerance), best over
   molecule spectra x up to 3 stored spectra per candidate.
4. Output 25 SMILES per molecule, best first.

Result: 0.088 public LB. Weak but our best hidden-test signal so far.
Why it works at all: exact mass cuts millions to hundreds; cosine is robust
and needs no training, so it survives the train/test chemistry shift.
