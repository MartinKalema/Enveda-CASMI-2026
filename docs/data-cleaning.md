# Data cleaning: audits, numbers, reusable module

All work lives in `v-clean/`; artifacts (gitignored via `*.parquet`) in `data-clean/`.
`data/train.parquet` / `data/test.parquet` were treated READ-ONLY throughout.
Headline: train is big but dirty in specific, library-dependent ways. A cheap,
evidence-based clean drops ~2% of spectra and aligns train/test peak densities.

## 1. Adduct-label audit (`adducts.py`, `audit_adducts.py`)

Method: neutral exact mass from `molecular_formula` (monoisotopic table) vs
precursor-derived expectation under the labelled adduct, tolerance
max(10 ppm, 0.01 Da). Generic compositional adduct parser covers 120/121
adduct types (only `[Cat]2+`, 15 rows, unparseable); verified byte-equal
against the hand table on all 21 common adducts.

- Auditable rows: 2,524,933/2,539,608 (99.42%). Rest: charged ion formulae
  (`C15H11O6+`, 11,753 rows, 0.46%).
- Label OK 98.02%, **mismatch 1.98% (50,013 rows)**.
- Mismatch per library: massbank **9.1%**, riken **6.6%**, gnps **6.1%**,
  mona 5.5%, drug_plus 3.5%, msdial 0.2%, pluskal/spectraverse/enveda ~0%.
- Worst adducts: `[M-2H2O+H]+` 18.4%, `[M+2H]2+` 11.1%, `[M]+` 9.5%,
  `[M-H2O+H]+` 7.6%, `[M+K]+` 5.3%.
- Only **8.3% of mismatches are rescued by an alternate same-polarity adduct**
  (top fix `[M+H]+`→`[M+Na]+`, 838 rows). The rest are wrong formula/structure
  or bad precursor mz, NOT simple H/Na swaps: the "gnps Na/H swap" hypothesis
  explains <0.4% of gnps rows. Do not relabel blindly.
- Bonus: 6,087 rows (0.24%) have adduct polarity contradicting
  `ionization_mode`, mostly riken `[M-H]-` under positive mode (4,870).
- `adduct` vs `adduct_orig` already differ on 183,785 rows (prior
  normalization); our audit is on the normalized label.

## 2. Label-quality filter (`audit_quality.py`, `clean.py`)

- `|precursor_error_ppm| > 20` excludes 4.7% overall; >10 excludes 6.9%.
- The column means different things per library: P(adduct-mismatch | err>20)
  is ~0.87-1.0 for gnps/massbank/mona/msdial/drug_plus/spectraverse but
  **0.0002 for pluskal_ms2** and 0.61 for riken. A blanket 20 ppm cut would
  trash 9.6% of pluskal spectra that are perfectly fine.
- Final rule (`clean.flag_frame`): drop spectra whose adduct label fails the
  mass check AND no alternate adduct fits, plus err>20ppm ONLY in the six
  trusted libs. **Drop rate 2.0%**, concentrated in massbank (9.3%), riken
  (6.8%), mona (6.3%), gnps (5.6%); enveda/pluskal untouched.

## 3. Peak denoising (`audit_peaks.py`)

- Train median 42 vs **test median 230 peaks** (means 158 vs 330).
- Per-library medians: riken/masaryk 9, massbank/msdial 11, pluskal 21,
  gnps 46, spectraverse 47, mona 57, drug_plus 108, enveda-180 138.
  Test looks like enveda-style dense spectra, not like riken/pluskal.
- Below 0.01 (max-normalized) intensity: 49% of train peaks, **82% of test
  peaks**. Test density is almost entirely low-intensity noise.
- Floor 0.01 + top-200 cap leaves median 13 (train) vs 18 (test) peaks,
  vs 42 vs 230 raw: distributions align. Mean self-cosine(raw, denoised)
  0.78, so ~3/4 of signal energy survives.
- Both train and test are already max-normalized (100% have max == 1.0).
- Recommendation: denoise BOTH query and reference with 0.01/top-200 before
  any cosine (`clean.clean_spectrum`). Never compare raw dense test spectra
  against sparse train spectra.

## 4. Train dedup (`audit_dedup.py`)

- **137,551 exact-duplicate rows (5.42%)** in 63,186 hash groups (rounded
  mz@4dp + intensity@3dp), mostly pairs (max group 139).
- 55,693 groups are within one structure; 7,493 span >1 inchikey14
  (stereoisomer/tautomer splits or true collisions: inspect before merging).
- 54,839 dup groups span >1 library: same spectrum re-ingested across libs.
- 9.2 spectra/structure (median 6); 6.2% of structures are singletons;
  10.2% of structures appear in >1 library.
- Same-structure same-adduct mean cosine **0.445** (median 0.373) vs
  cross-adduct **0.058**: pool/score within adduct, or expect near-zero
  cross-adduct matches. Retrieval indexes should keep one rep per
  (structure, adduct), not per spectrum.

## 5. Formula-mass index (`probe.py`)

- 51,291 unique formulae, 98.99% parsed (rest: charged ion formulae).
- **65.3% of formulae have a 10 ppm neighbour**; 1,184/1,556 nominal masses
  are shared (max 289 formulae per nominal mass). Precursor mass alone
  cannot pick a formula: always combine with spectral evidence.
- Artifact: `data-clean/formula_mass_index.parquet` (exact-mass keyed).

## 6. 20-query retrieval probe (`probe.py`, pool 2000, structure-disjoint)

- Top-1: raw 4/20 = denoised 4/20. Top-5: **5/20 -> 7/20**.
- Median rank of truth spectrum: **30 -> 16**. Mean truth cosine 0.22 -> 0.28.
- 4/20 queries unmatchable either way (truth cosine 0.0: no shared peaks).
- All 20 sampled queries happened to carry good adduct labels, so the probe
  does not measure the correction path; the correction's value is in §1-2
  (neutral-mass windows for candidate gating, not ranking).

## Reuse

```python
from v-clean.clean import flag_frame, clean_train, clean_spectrum, correct_adduct
flags = flag_frame(train_df)   # label_ok / corrected_adduct / rescued / drop
clean = clean_train(train_df)  # + adduct_used, neutral_used, dn_mzs/dn_ints/dn_n
```

Artifacts: `data-clean/adduct_audit.parquet` (per-row residual + best-fit),
`quality_flags.parquet`, `spec_hash.parquet`, `formula_mass_index.parquet`,
`probe20.csv`. Regenerate with `audit_*.py` / `probe.py` (read-only on `data/`).
