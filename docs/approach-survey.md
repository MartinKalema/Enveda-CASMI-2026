# Approach survey: practical winning methods for MS/MS structure retrieval (MRR@25, offline 9h kernel)

Branch: feat/domain-research | Date: 2026-09-21 | Scoreboard: v0 0.088 vs top 0.409
Context: Kaggle Enveda CASMI26 molecule-id (mass-filtered candidate retrieval, up to 25 SMILES per
molecule, MRR@25). Kernel runs offline, ~9h CPU/GPU, no internet. Test = MCES-novel natural
products; train is 99% out-of-domain. Ranked below by expected MRR@25 value per implementation cost.

## Rank 1. MIST fingerprint retrieval (reimplement, don't just download)
How it works: each MS/MS peak is annotated with a chemically plausible subformula (mass error
<= ~15 ppm, RDBE filter), turning a spectrum into a *set* of (subformula, intensity, mass-error)
tokens instead of raw m/z bins. A set transformer ("Formula Transformer") embeds that set plus
precursor/adduct/instrument covariates into a vector trained to match the molecule's Morgan
fingerprint; at test time you predict the fingerprint and cosine-rank same-mass candidates.
MassSpecGym retrieval: MIST hit@1 14.6% vs 5.2% DeepSets vs 2.5% FFN — the whole gap comes from
subformula labelling, which is exactly our v1 plan in constraints-and-literature.md.
Offline verdict: YES, ideal. Code MIT-licensed (github.com/samgoldman97/mist), pretrained weights
on Zenodo, trains <3h on one GPU, inference is NumPy + transformer forward + FAISS. No license
server, no internet needed.
Expected value: HIGHEST. This is the backbone. Own reimplementation trained on competition data
(all 1-9 spectra/molecule with attention/mean fusion, energy-based loss over same-mass decoys)
is the most likely single jump from 0.088 toward 0.20+.

## Rank 2. MIST-CF formula ranking + FastFilter funnel
How it works: same subformula-set encoder as MIST, but the target is the precursor's chemical
formula, trained with an energy-based softmax over the true formula plus hard decoy formulae at
similar mass. Predicted formula prunes candidates to isomers (orders-of-magnitude cut); a cheap
formula prior keeps top-256 candidates at ~99% recall and only those go to the expensive scorer.
Tied-winning CASMI2022 formula entry (0.868 joint accuracy with SIRIUS, zero curation, ~1/3 the
runtime); standalone 0.769 top-1 on NPLIB1+NIST20.
Offline verdict: YES. Same repo family (mist-cf, MIT), same weights-via-Zenodo pattern, same
subformula labeller shared with Rank 1 — nearly free once MIST is built.
Expected value: HIGH. Formula errors cascade into structure misses (CASMI median formula
accuracy only ~71%), so a good formula stage buys both recall (right isomers in the 25) and
precision (fewer wasted slots). Build jointly with Rank 1.

## Rank 3. Forward-simulation rerank (ICEBERG / FraGNNet via ms-pred + MassSpecGym weights)
How it works: instead of spectrum->structure, go structure->spectrum: a graph neural network
predicts the theoretical MS/MS of each candidate (FraGNNet = combinatorial fragmentation +
GNN-learned breakage probabilities; ICEBERG = two-stage generate-then-score fragment model),
then rerank candidates by predicted-vs-observed spectral similarity. Physics-grounded, so it
generalises to novel scaffolds better than memorised fingerprints.
Offline verdict: YES, with care. FraGNNet (github.com/FraGNNet/fragnNet) and ICEBERG through
coleygroup/ms-pred are open source with pretrained MassSpecGym weights freely downloadable —
bundle weights into the kernel image before going offline. Cost is per-candidate simulation, so
only rerank the top-50..256 from the Rank 1/2 funnel, precompute and cache.
Expected value: HIGH as a reranker, LOW as a retriever. Best used as 1-2 LightGBM features
(predicted-spectrum cosine, explained-intensity fraction) on top of MIST scores. Likely the
rank-1 tiebreaker that converts rank 2-5 into rank 1 (MRR 0.5 -> 1.0).

## Rank 4. SIRIUS / CSI:FingerID (learn from it, do not run it)
How it works: SIRIUS builds a fragmentation tree (each peak gets a subformula, edges are neutral
losses, tree scored by isotope + mass error + loss plausibility), derives the formula from the
best tree, then CSI:FingerID predicts ~thousands of fingerprint bits with per-bit SVMs trained on
tree kernels, and searches structure DBs by predicted fingerprint. The reference standard for a
decade; still top-tier on CASMI formula/structure tasks.
Offline verdict: NO for the kernel. CSI:FingerID fingerprint prediction runs as a web service
requiring an account/license (free for academic use but needs login + internet); full SIRIUS CLI
needs the same. Cannot run in a 9h offline Kaggle kernel. Implementations of the *ideas*
(fragmentation trees, loss features, per-bit fingerprint classifiers) are fair game and already
folded into Ranks 1-3.
Expected value: MEDIUM as inspiration, ZERO as a dependency. Copy its features (neutral-loss
counts, tree score, CANOPUS compound-class приходится as a rerank prior), never call it.

## Rank 5. MS2DeepScore / Spec2Vec (analog-search features)
How it works: both learn spectrum embeddings from large libraries without structures at training
time. Spec2Vec ports Word2Vec to spectra (peaks/losses = words, co-occurrence = chemistry) and
scores pairs by embedding cosine; MS2DeepScore trains a Siamese network to directly regress the
Tanimoto structural similarity of the underlying molecules from two spectra. They beat raw cosine
at finding *analogs* (same scaffold, different substituents).
Offline verdict: YES, easily. Both are pip-installable Apache-2.0 packages (matchms ecosystem:
`pip install spec2vec`, ms2deepscore on GitHub) with pretrained models downloadable ahead of
time; pure-Python/PyTorch inference, no license server.
Expected value: MEDIUM-LOW as primary ranker (they retrieve analogs, and MRR pays only for the
exact structure), MEDIUM as features: max/mean MS2DeepScore of a candidate's near-neighbors in
the train library, Spec2Vec library-match score, scaffold-consistency vote. Cheap to add once
the pipeline exists; expect +0.005..0.02, mostly in ranks 5-25.

## Rank 6. MetFID (cheap CNN fingerprint baseline / ensemble member)
How it works: bins the MS/MS into a fixed vector, runs a small convolutional/dense network to
predict the molecule's fingerprint bits, ranks same-mass candidates by fingerprint cosine. The
pre-transformer generation of learned fingerprint retrieval (2020-2022), solid but no peak-level
chemical features.
Offline verdict: YES. Python package, open, trains fast on CPU/GPU, fully offline.
Expected value: LOW-MEDIUM. Strictly superseded by MIST (MassSpecGym FFN tier ~2.5% hit@1 vs
MIST 14.6%), but trains in minutes and its errors are uncorrelated with transformer errors, so
it earns a slot as an ensemble/rerank feature and as the fallback if the MIST reimplementation
slips schedule. Build only after Ranks 1-3 work.

## Rank 7. MSNovelist (de novo slot-filler, not a ranker)
How it works: takes a SIRIUS formula + CSI:FingerID predicted fingerprint and decodes novel
SMILES with a fingerprint-to-structure RNN (encoder-decoder trained on millions of structures),
then rescores generated candidates against the spectrum. Can propose structures absent from every
database — the only method here that handles true "dark matter".
Offline verdict: NO as published (inherits the SIRIUS/CSI:FingerID online dependency), and de
novo top-1 accuracy is 0.00 for all MassSpecGym baselines — exact generation is unsolved.
A home-grown variant (our own fingerprint->SMILES decoder) is offline-legal but expensive.
Expected value: LOW for MRR (generated structures almost never hit rank 1), OPTIONAL for slots
16-25: if retrieval confidence is low, spending leftover slots on diverse de novo guesses costs
nothing (MRR only rewards hits). Park until retrieval plateaus; MS2Mol-style Enveda prior art
suggests hosts respect it, but it will not move the needle first.

## Rank 8. "ESP" (no canonical method — treat as the embedding-retrieval pattern)
How it works: a web search finds no single dominant "ESP" model for MS/MS structure retrieval
(likely confusion with Spec2Mol's spectral encoder, ICEBERG spectrum prediction, or generic
"embedding + spectral prediction" pipelines). The underlying pattern — contrastive
spectrum<->molecule joint embedding, retrieve scaffold then refine with a graph decoder — is real
and covered by Ranks 1, 3, and 5.
Offline verdict: n/a — nothing concrete to install. Do not chase the acronym; invest in Rank 1
contrastive training (spectrum-to-fingerprint is itself a joint-embedding model).
Expected value: NONE beyond what Ranks 1/3/5 already capture.

## Rank 9. CASMI 2022/2024 winning tricks worth stealing
- Mad Hatter (CASMI2022 meta-winner): ensemble many DBs/tools and rerank by consensus + metadata
  (citation counts, DB presence). Lesson: library-prior and cross-tool agreement features are
  free MRR — add candidate prior (train/COCONUT/PubChem presence) to the reranker.
- Joint SIRIUS + MIST-CF formula (0.868 CASMI2022): two independent formula predictors agree =
  high confidence; disagree = hedge with both formulae's isomers across slots.
- ZODIAC (SIRIUS companion): Gibbs-sampling formula re-estimation over joint LC-MS runs — our
  analogue is per-molecule multi-spectra fusion (1-9 spectra vote on one formula/structure).
- COSMIC/CANOPUS: confidence scores and compound-class predictions as rerank priors; class match
  (e.g. predicted flavonoid vs candidate flavonoid) is a cheap, generalising feature.
- Universal CASMI lesson: winners dedup tautomers/stereoisomers (InChIKey-first-block) because
  duplicate slots are wasted slots — already our Slot-economy rule.

## Rank 10. Public Kaggle notebook ideas for THIS competition (enveda-CASMI26-molecule-id)
Observed patterns in shared notebooks/discussion (code tab, "Tautomer Dedup" threads, CASMI
2012/2016 648-spectra subset): modified-cosine baselines (~our v0 tier), InChIKey-14 dedup to
25, mass-windowed candidate prefiltering, RDKit Morgan fingerprints for scoring. These set the
floor (~0.09-0.15), not the ceiling — beat them with learned fingerprints (Rank 1), not better
cosine.
Offline verdict: YES — notebooks are the source of data loaders, dedup snippets, and candidate
DB handling; vendor nothing learned that needs internet at inference.
Expected value: LOW for MRR directly, HIGH for velocity: fork the best data-loading/dedup
notebook instead of rewriting parsing, and spend saved days on Ranks 1-3.

## Bottom line (build order)
1. MIST-style subformula fingerprint retrieval + MIST-CF formula funnel (Ranks 1-2) — the game.
2. ICEBERG/FraGNNet simulation rerank on top-256 (Rank 3) — the tiebreaker.
3. Library-prior + analog + class features, LightGBM rerank, InChIKey-14 dedup (Ranks 5/6/9/10).
4. Skip SIRIUS/CSI:FingerID/MSNovelist as runtime dependencies (Ranks 4/7); revisit de novo
   slot-fillers only after retrieval plateaus.
