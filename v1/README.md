# v1: subformula explained-intensity (LB 0.044 - worse than v0)

Plain English: instead of comparing raw peaks, translate each mystery peak
into a chemical subformula (e.g. this fragment is C6H5O), then score candidates
by how much of the mystery signal their formula can explain.

Steps:
1. Subformula labeller: enumerate all subformulae of a candidate formula
   (RDBE-filtered), match peaks within 15ppm. True formulae explain ~72% of
   signal vs ~26% for wrong ones.
2. Rank mass-window candidates by explained intensity + formula frequency prior.
3. Fuse multiple spectra per molecule by max score.

Result: 0.044 LB - WORSE than v0. Lesson: on truly novel molecules the
explainer is confidently wrong; it overfits known chemistry. Kept as a
possible auxiliary feature, not the main ranker.
Validation (structure-disjoint, n=30): MRR 0.099 - decent locally, didn't transfer.
