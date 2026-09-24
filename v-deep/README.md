# v-deep: pretrained-codebase shootout (starting with MS2DeepScore)

Plain English: instead of our hand-built scorers, use the published
pretrained models as drop-in channels on identical frozen queries. First
bout: MS2DeepScore (397MB Siamese weights) vs our cosine. Same disjoint
protocol as the metfrag validation so numbers compare directly.

## Bout 1: MS2DeepScore (pretrained 397MB Siamese) vs cosine, n=30 disjoint
Result: deep MRR 0.041 vs cosine 0.058, corr 0.348. VERDICT: loses as floor
replacement (it predicts structural similarity for analogs, not exact-rank).
Moderate correlation leaves a possible analog-recall arm role, unproven.
