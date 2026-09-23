# v9: GBM-25 fills (pending)

Plain English: v8's fingerprint fills plateaued like every fill before it.
v9 keeps the cosine top-5 floor and orders fills with a calibrated ranker
trained on 6 features (cosine, entropy, analog, mass error, formula prior,
Tanimoto-to-top3) instead of one hand-picked signal. GBM-25 beat the
3-feature ranker on held-out seeds (0.134/0.168 vs 0.062).
