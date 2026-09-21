# v6: GBM rerank below a cosine floor (pending)

Plain English: v5 validation showed a calibrated ranker beats hand-mixing on
every held-out seed (0.061-0.098 vs 0.002-0.007). v6 keeps v0-cosine top-5
untouchable (rule 26) and orders everything below by GBM P(answer) over
cosine/entropy/analog features. Ranker ships as a 266KB pickle in the
fingerprint dataset; kernel needs sklearn (standard Kaggle image).
