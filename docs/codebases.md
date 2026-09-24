# Codebase + weights inventory: what to reuse vs rewrite

Scope: Kaggle MS/MS retrieval, offline 9h CPU/GPU kernel, no internet at scoring.
Winners must be MIT open-source. Rule used below: anything GPL/AGPL/NC-gated or
internet-gated is design-donor-only, never in the scoring path.

Pin rule: vendor with `git clone --depth 1 --branch <branch>` on the vendor date,
record `git rev-parse HEAD`, and bundle as a tarball. Tags below are what to pin;
exact SHAs must be captured at vendor time (not fabricated here).

## Summary table

| # | Item | Repo (pin) | License | Weights (size, license) | Lang / deps | Offline? GPU? 400 mols? | Reuse vs rewrite |
|---|---|---|---|---|---|---|---|
| 1 | MIST (fp + contrastive) | `github.com/samgoldman97/mist`, branch `main_v2` | MIT (`LICENSE.md`) | `quickstart/00_download_models.sh` → Zenodo `15122968` (`mist_msg.pt` etc, ~100–300 MB total), MIT | Python/PyTorch, RDKit, MAGMa/SIRIUS for labeling | Offline yes after vendoring; GPU for train, CPU infer OK (~10–30 s/mol batched) | REUSE weights + `src/mist` subformula labeller; RETRAIN optional (our FPNet bit-space differs) |
| 2 | MIST-CF (formula) | `github.com/samgoldman97/mist-cf`, `main` | MIT (same org; verify LICENSE at vendor) | checkpoints via repo releases / Zenodo linked in README (~100–200 MB) | Python/PyTorch | Offline yes; CPU infer OK | REUSE formula ranker as prefilter + GBM feature; do NOT reimplement energy model |
| 3 | MassSpecGym (benchmark, not a model) | `github.com/pluskal-lab/MassSpecGym`, tag `v1.3.1` (2025-03-23) | MIT code + MIT data (HF `roman-bushuiev/MassSpecGym`) | data via HuggingFace (~GBs; use only splits/eval code, ~MBs) | Python/PyTorch-Lightning | Offline yes | REUSE split protocol (MCES/InChIKey-disjoint) + eval metrics; do NOT ship dataset in kernel |
| 4 | ms-pred / ICEBERG (forward sim) | `github.com/coleygroup/ms-pred`, branch `main` (or `iceberg_2.0` / `iceberg_1.0` for paper parity) | MIT code | Open: MassSpecGym-trained via Dropbox link in README (100s MB). NIST20/23-trained: gated, needs NIST license + email maintainer — DO NOT USE | Python/PyTorch + DGL/torch-geometric, RDKit, MAGMa labels | Offline yes once vendored; train needs GPU (~6h A5000); CPU infer feasible | REUSE open MSGym weights OFFLINE-DISTILL only: precompute fragment bags per pool structure → ship `.npy`, lookup in-kernel. Never infer in-kernel |
| 5 | FraGNNet | `github.com/FraGNNet/fragnnet`, `main` | BSD-2-Clause | checkpoints in repo releases (100s MB) | Python/PyTorch-Geometric | Offline yes; GPU train, CPU lookup | Same as ICEBERG: REUSE as second distillation source; no in-kernel inference |
| 6 | matchms (+ flash entropy) | `github.com/matchms/matchms`, latest release; entropy: `github.com/YuanyueLi/MSEntropy` (`ms_entropy` pip) + `YuanyueLi/FlashEntropySearch` demo | Apache-2.0 (both) | no weights (algorithmic) | Python/NumPy/Numba | Offline yes; CPU ms/spectrum | REUSE `ms_entropy` if wheel vendored, else REWRITE 30-line entropy kernel in numba (already done in fork-034 cell 3). Copy Flash matched-only short-circuit + hybrid modes as logic, not dependency |
| 7 | Spec2Vec | `github.com/iomega/spec2vec` (model) + `matchms` scoring | Apache-2.0 | Pretrained GNPS model via spec2vec docs (~10s MB, Apache); else train gensim Word2Vec on train.parquet ~40 min CPU | Python/gensim | Offline yes; CPU 0.14 s/query | REUSE pretrained embeddings as recall arm only (analog prefilter + 1 ranker feature); REWRITE nothing; gate by peak count |
| 8 | MS2DeepScore | `github.com/matchms/ms2deepscore` | Apache-2.0 | `ms2deepscore_model.pt` via Zenodo (>500k spectra GNPS/MoNA/MassBank/MSnLib, ~10s MB), Apache | Python/TensorFlow or PyTorch (version-dependent) | Offline yes; CPU embed precomputable | REUSE model as analog pre-ranker embedding; do NOT use for Channel-1 identity |
| 9 | MS2Query | `github.com/iomega/ms2query` | Apache-2.0 | Pretrained GNPS library + RF (`ms2query` data download, 100s MB–GBs; needs vendoring) | Python/sklearn + Spec2Vec + MS2DeepScore | Offline yes if library vendored; 80 spectra/min laptop | REUSE two ideas, not the package: kNN-mean over top-10 + neighbor-consensus features. Running full MS2Query in-kernel is overweight |
| 10 | SIRIUS + CSI:FingerID | `github.com/sirius-ms/sirius` (Java GUI+CLI v6.x) | Client AGPL/GPL-family; web services need account; academic-free, commercial via Bright Giant | Fingerprint predictor lives server-side (no redistributable weights) | Java; needs login + license + webservice URLs | Offline NO (requires internet + account). Per-spectrum fragmentation-tree cost too high | DO NOT RUN. Design donor only (fragmentation-tree idea already covered by MIST-CF). FLAG: AGPL + account-gated + internet-gated = triple landmine |
| 11 | BUDDY / msbuddy | `github.com/Philipbear/msbuddy` (pip `msbuddy`) + `Philipbear/BUDDY_Metabolomics`; paper repo `HuanLab/BUDDY` | Apache-2.0 (`msbuddy` docs) / MIT (metabolomics repo) | No weights (LightGBM + 3.5M subformula DB bundled, ~100s MB) | Python/LightGBM, RDKit | Offline yes; CPU feasible | REUSE bottom-up stitching logic + LightGBM-on-38-features + Platt-FDR shape for our formula reranker; vendoring whole package optional |
| 12 | MIST + MolForge (de novo fills) | MIST (above, MIT) + `github.com/knu-lcbc/MolForge` + repro `harrylaucngd/MIST-MolForge` | MolForge CC BY-NC 4.0 — LANDMINE (non-commercial) | MIST `mist_msg.pt` (MIT) + MolForge ckpt via Google Drive/OSF mirror (100s MB, NC) | Python/PyTorch | Offline yes; CPU ~30–60 s/mol; batch-size-1 (mask bug) | REUSE for slots 16–25 ONLY, severable behind `--no-denovo`. Never in ranks 1–15 |
| 13 | MS-BART | `github.com/OpenDFM/MS-BART` | NO license declared — FLAG (treat as all-rights-reserved until clarified) | Figshare bundle: 4M-pretrain + CANOPUS/MSGym finetunes (GBs) | Python/PyTorch | Offline yes if vendored; CPU OK | DO NOT RUN until license + weights verified. Steal SELFIES-validity + fp-token ideas only |
| 14 | DiffMS | `github.com/coleygroup/DiffMS` (adapted from DiGress) | MIT | Pretrained encoder/decoder via `general_default.yaml:load_weights` (100s MB) | Python/PyTorch + torch-geometric, CUDA 11.8 | Train GPU; CPU sampling heavy (100s of diffusion steps) | REUSE as second-chance generator with reduced steps, additive to MolForge. MIT-clean |
| 15 | DreaMS (embeddings) | `github.com/pluskal-lab/DreaMS` | MIT | Zenodo `10997887` pretrained (100s MB, MIT); GeMS data on HF | Python/PyTorch 3.11 | Offline yes; CPU embed OK | REUSE embeddings as analog-channel feature (already tested: no-win per finding 11 — keep as fallback only) |
| 16 | FLARE | Paper bioRxiv `10.64898/2026.01.27.702086` (Chen 2026); no public repo found at survey time | Unknown — FLAG until repo appears | none | PyTorch/GNN (assumed) | n/a | REWRITE the idea (bidirectional peak↔atom MaxSim head) on our encoder; do not wait for code |
| 17 | JESTR | `github.com/HassounLab/JESTR1` | LICENSE file present (verify; some datasets license-gated, NIST parts not redistributable) | NPLIB1 weights released; other datasets gated | Python/PyTorch, GPU A100 train | Offline yes for NPLIB1 weights; CPU infer OK | REUSE loss recipe (CMC/InfoNCE τ=0.05 + late hard-decoy regularization) + ranker for generated candidates; retrain on our data |
| 18 | MVP (multi-view contrastive, Anal Chem 2026) | No PDF/repo (paywalled, cited via FLARE Table 1) | Unknown | none | n/a | n/a | Idea only: multi-view InfoNCE. No code to reuse |
| 19 | CatBoost (YetiRank/YetiLoss/QueryRMSE) | `github.com/catboost/catboost` (pip `catboost`) | Apache-2.0 | no weights | C++/Python, CPU | Offline yes (wheel vendored); 400 queries = seconds–minutes | REUSE: primary listwise upgrade (YetiLoss-MRR + QueryRMSE anchor). Mandated GBDT implementation |
| 20 | LightGBM (lambdarank) | `github.com/microsoft/LightGBM` (pip `lightgbm`) | MIT | no weights | C++/Python, CPU | Offline yes | REUSE as diversity view (LambdaMART) in stack; demoted below CatBoost |
| 21 | mzMine (IIMN, analog modes) | `github.com/mzmine/mzmine3` | LGPL/GPL-family — FLAG for MIT kernel (linking/copyleft) | no weights | Java | Offline yes but Java in kernel is overweight | DO NOT BUNDLE. Reuse ideas only (adduct-grouping abstraction, modified-cosine/cosine-no-precursor/MS2DeepScore modes) |
| 22 | NPClassifier | `github.com/mwang87/NP-Classifier` (also GNPS endpoint) | MIT | weights bundled (~MBs) | Python/TensorFlow | Offline yes (local model) but needs internet if via GNPS API | REUSE local MIT model for class features / evaluation only; never call API in-kernel |
| 23 | ClassyFire | API `classyfire.wishartlab.com`; wrappers `earth-metabolome-initiative/classyfire`, `aberHRML/classyfireR` | Data/service: free non-commercial, commercial needs permission — FLAG | no redistributable weights (rule-based taxonomy ChemOnt) | Web API (needs internet) | Offline NO | DO NOT CALL in kernel. Precompute offline or skip; MolNetEnhancer-style family consensus is the portable idea |

## Per-item reuse verdicts (plain English)

1. **MIST.** Best fingerprinter we can legally run. Take their weights and their
   peak-labeling code. Skip retraining unless we commit a GPU session.
2. **MIST-CF.** Best formula gate. Take it as a filter plus a ranker feature.
3. **MassSpecGym.** Not a model. Copy how they split data (no near-duplicate
   leakage) and how they score. Don't put their data in the kernel.
4. **ICEBERG (ms-pred).** Good physics simulator, too slow live. Run it once
   offline, save fragment lists, look them up in the kernel. Use only the open
   MassSpecGym weights; the better NIST weights need a paid license.
5. **FraGNNet.** Same deal as ICEBERG, slightly better numbers. Second source
   of offline fragment lists.
6. **matchms / flash entropy.** Our numba kernel already is this. Copy the
   speed trick (skip pairs with no shared peaks) and the hybrid fragment-or-loss
   mode. Vendoring the pip package is optional.
7. **Spec2Vec.** Cheap similarity that finds relatives. Use the ready-made
   model as an extra recall net for analogs, not for exact matches.
8. **MS2DeepScore.** Same as Spec2Vec but stronger and heavier. Use its
   ready-made model to pre-rank analogs offline.
9. **MS2Query.** Don't run the whole tool. Copy its two tricks: average over
   the top-10 neighbors instead of trusting the single best, and score a
   candidate by how its neighbors score.
10. **SIRIUS / CSI:FingerID.** Cannot run offline, needs an account, and the
    license is copyleft plus commercial-gated. Read the papers, don't ship it.
11. **BUDDY.** Good formula method with a friendly license. Copy how it builds
    formulas from fragments upward and how it calibrates confidence. Runs offline.
12. **MIST+MolForge.** Best lottery ticket for novel molecules (slots 16–25).
    Works, but the decoder is non-commercial licensed, so keep it behind a flag
    we can turn off and never let it touch ranks 1–15.
13. **MS-BART.** Promising but no license stated and weights live on Figshare.
    Don't touch until both are confirmed. Ideas only.
14. **DiffMS.** Clean MIT generator. Use sparingly as a backup generator with
    fewer steps so the CPU survives.
15. **DreaMS.** Clean MIT embeddings. Already tested with no win here; keep as
    a fallback feature, not a channel.
16. **FLARE.** Best new retrieval idea (match peaks to atoms both ways) but no
    code out. Rebuild the small scoring head ourselves.
17. **JESTR.** Good training recipe (contrastive + hard same-formula decoys at
    the end). Reuse the loss schedule and its ranker for generated candidates.
18. **MVP.** No paper or code available. Copy the one-line idea (train on
    multiple views of spectra and molecules), nothing else.
19. **CatBoost/YetiRank.** Our ranker upgrade. Friendly license, CPU-only,
    made for tiny query counts. Adopt.
20. **LightGBM-rank.** Second ranker for variety in the stack. Friendly license.
    Use for the LambdaMART view only.
21. **mzMine.** Java toolbox with copyleft license. Don't bundle; copy the
    adduct-grouping idea in our own Python.
22. **NPClassifier.** Friendly MIT classifier. Run the local model for analysis
    features, never the web API inside the kernel.
23. **ClassyFire.** Web-only classification with a non-commercial data license.
    Never call from the kernel; precompute or skip.
