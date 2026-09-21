# v2: fingerprints (blend MRR 0.047, learned-fp MRR 0.038 - both below v1)

Plain English: convert structures into 2048-bit fingerprints (which chemical
building blocks are present), predict the mystery fingerprint from its spectra,
rank candidates by fingerprint overlap (Tanimoto).

Tried:
1. Memory blend: average fingerprints of spectral neighbours. MRR 0.047.
   Raw-cosine neighbours are structurally too weak - matches the literature
   (learned MIST hit@1 14.6 vs plain FFN 2.5).
2. Learned MLP (12k-bin spectrum -> 2048 bits, torch/MPS, 200K spectra,
   structure-disjoint): val Tanimoto 0.21, retrieval MRR 0.038 after 5 epochs.
   Undertrained; may revisit with more epochs + subformula inputs.

Lesson: ranking isn't the cap - candidate recall is. v3 expands candidates
with COCONUT (627K new natural products) under v0-cosine ranking.
