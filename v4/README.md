# v4: analog propagation + entropy (LB 0.013 - worst)

Plain English: score every candidate (even ones with no lab spectra, like
COCONUT) through lookalikes that DO have spectra: take train spectra that
look like the mystery (entropy similarity, within 200 Da), weight by
similarity^3 x fingerprint overlap. Plus entropy similarity as a direct
channel. Local disjoint MRR 0.067 (analog) vs 0.018 (entropy) vs 0.002 cosine.

Result: 0.013 LB. Post-mortem: the validation split was uncalibrated
(cosine anchor 0.002 there vs 0.099 elsewhere), and v4 replaced the proven
cosine top-5 with entropy. Two errors: trusted one noisy split, and replaced
the floor instead of adding below it. Rules 21-27 in lessons-learned.
