# Enveda CASMI 2026 - Deep Dive: Goal, Strategy, Capabilities
Date: 2026-09-21 | Project: enveda-casmi26-molecule-id

## 1. Competition in one paragraph
Predict 2D structures (SMILES) of small molecules from LC-MS/MS spectra. Test: ~400 anonymous natural-product-like molecules (plant/mammal/microbe, confirmed/hypothesised NPs, analogs, plausible synthetics), ~1213 spectra in preview (~1500 hidden on rerun, 1-9 spectra/molecule, median 3, mass 157-1159 Da median 348, all Bruker timsTOF). For each `molecule_id` submit up to 25 SMILES ranked best-first, semicolon-joined. Featured Code Competition, $50k (16/12/9/7/6k), 1087 teams, deadline 2026-12-14, 5 subs/day, 2 finals, MIT open-source, Notebook-only rerun. Metric: MRR@25. Current top: 0.409, 0.388, 0.387 - low absolute = hard, rank gaps small.

## 2. Exact task mechanics (from Data page + MCP `get_competition`)
- `test.parquet` per spectrum: `molecule_id, spectrum_id, ms2_mzs[], ms2_normalized_intensities[] (base=1.0), base_peak_intensity (higher=less noise), adduct (7 values, 79% [M+H]+, rest [M-H]-, [M+Na]+, [M+CH2O2-H]-, [M+NH4]+, [M+K]+, [M+Cl]-), ionization_mode, instrument_type=timsTOF always, precursor_mz, collision_energy_ev[] ([20] single vs [20,40,60] merged), collision_energy_orig/units=eV`.
- `train.parquet` 2,539,608 x 18, 277,566 smiles / 275,950 inchikeys: label `normalized_smiles` (RDKit-standardised), `inchikey/inchikey14` (14=stereo-independent skeleton), `molecular_formula`, `ingest_lib`, `precursor_error_ppm` (uncleaned = label-quality signal), `num_peaks` (median 42, mean 158, max 73k - heavy tail), plus same spectral cols except `instrument_type` noisy free-text, `base_peak_intensity` null when pre-normalised.
- Library split (domain shift is the crux):
  - enveda-180 1,153,785 / 182,941 - same timsTOF, consistent, but synthetic drug-like, wrong chemistry
  - pluskal_ms2 527,581 / 46,821 - Orbitrap, consistent multi-CE protocol
  - riken 347,171 / 15,892 - plant specialised metabolites (chemistry-closest large)
  - gnps 220,849 / 45,750 - community NPs, heterogeneous
  - massbank/mona/spectraverse/msdial - mixed QTOF/Orbitrap
  - enveda-np-examples 1,184 / 250 - ONLY in-domain (same instrument+pipeline, common NPs, overlaps other libs for comparison)
- `collision_energy_ev` is best-effort eV conversion (NCE via Thermo nominal formula, approximate). Use it, not orig.
- Hidden test replaces preview file, same size. Public LB 33%, private 67%.
- Code limits: CPU/GPU <=9h, internet OFF, public external data + pretrained allowed, `submission.csv` in /kaggle/working, strict validation (missing cols/nulls/dup molecule_id/>25 guesses = reject). Winners must deliver training+inference code.

## 3. Metric: MRR@25 (not MAP)
`MRR = 1/U * sum(1/rank_u)`, rank of FIRST correct in your 25, 0 if miss. rank1=1.0, rank2=0.5, rank25=0.04. Only first correct counts. No penalty for wrong beyond slot occupied, <25 allowed. Implication: optimise recall@25 AND rank1 precision. Tautomer/duplicate SMILES waste slots - dedup by InChIKey14 mandatory. Public notebooks already hint: "Four-Channel Ranker, Tautomer Dedup".

## 4. Winning strategy (retrieval + rerank, not de novo)
1. **Candidate generation by mass:** precursor_mz + adduct -> neutral mass (+/- 10-20ppm) -> filter giant DB (train 277k + COCONUT/LOTUS NP + PubChem). This reduces millions to 100s-1000s. Formula inference from exact mass + isotope helps.
2. **Spectral scorer (learned):** peak-set Transformer (mz sinusoidal + intensity, CE/adduct/mode embeddings) -> predict Morgan fingerprint or joint embedding (contrastive MS2<->structure, cf. MIST/MS2Mol/MassSpecGym). Train on all libs, upweight timsTOF + NP-like (riken/gnps/np-examples), condition on `collision_energy_ev`.
3. **Multi-spectra fusion per molecule:** 1-9 spectra, different CE/adducts. Aggregate (attention/mean/max, weight by `base_peak_intensity`), adduct-aware. Predict per molecule, not per spectrum.
4. **Reranker (GBM + rules):** features = neural score, mass error ppm, fingerprint Tanimoto, neutral-loss matches, library prior, NP-likeness, instrument compatibility. Calibrate for MRR (rank1).
5. **Chemistry cleanup:** RDKit canonicalise, strip salts, standardise tautomers, dedup by inchikey14, fill to 25 with diverse analogs (mass neighbours). Never submit `CCO` placeholders.
6. **Domain adaptation:** fine-tune on enveda-np-examples (tiny but gold), pseudo-label, instrument-transfer (Orbitrap->timsTOF). Filter train by `|precursor_error_ppm|>20` (mislabeled).
7. **Code-compiled inference:** precompute candidate embeddings offline, FAISS, quantised model, batch, <9h, no internet. All assets in notebook.

Why this wins: test molecules are unseen NPs (novel), so memorisation fails; mass + fragmentation + NP prior + multi-spectra fusion generalises. Top 0.409 shows headroom - 0.02 gain = places.

## 5. Capabilities we need
- Data: pyarrow/pandas chunked 2.8GB, matchms spectrum processing, FAISS candidate index
- Chem: RDKit (canonical SMILES, InChIKey, formula, tautomer), COCONUT/LOTUS DBs
- ML: PyTorch (MPS locally, CUDA on Kaggle), Transformers, contrastive learning, LightGBM rerank, sklearn metrics (MRR@25)
- MLOps: Kaggle notebooks offline, `uv`, 9h budget profiling
- Already have: M4 Pro 12CPU/24GB/16-core Metal (dev, MPS), Kaggle MCP headless (Bearer KGAT verified `tools/list`), 1087-team LB tracking via `get_competition_leaderboard`

## 6. Knowledge/expertise required
- Mass spec: MS/MS fragmentation, adducts ([M+H]+/[M+Na]+/[M-H]-...), CE (eV vs NCE), precursor mass defect, noise vs base_peak_intensity, timsTOF vs Orbitrap/QTOF bias
- Cheminformatics: SMILES/InChI/InChIKey14, tautomerism, stereochemistry (evaluation is 2D - stereo-insensitive, use inchikey14), NP chemical space vs synthetic drug-like
- ML/IR: learning-to-rank, MRR optimisation, dense retrieval, fingerprint prediction, multi-instance learning (spectra->molecule)
- Competition: LB shakeup (33/67 split), 5/day discipline, 2-final selection, MIT code release

Next: build EDA notebook (mass/adduct/CE distributions) -> baseline mass-filter + cosine rerank -> neural fingerprint model.
