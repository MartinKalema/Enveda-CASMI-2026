# Enveda CASMI 2026 - Molecule ID From Mass Spectra

Local project for Kaggle challenge. Headless Kaggle MCP ready (no UI).

## Layout
- `data/` - train.parquet (2.5M rows, 2.8GB), test.parquet (1213 rows), sample_submission.csv
- `docs/kaggle-mcp-no-ui.md` - full research for no-UI MCP interaction
- `opencode.jsonc` - pre-wired `kaggle-official` (remote + KGAT token) and `kaggle-local` (uvx + username/key)

## Data schema (verified 2026-09-21)
- train: 2539608 x 18 - `normalized_smiles, inchikey, molecular_formula, ionization_mode, instrument_type, adduct, precursor_mz, ms2_mzs, ms2_normalized_intensities, num_peaks, ...`
- test: 1213 x 12 - `molecule_id, spectrum_id, ms2_mzs, ... precursor_mz, ...`
- sample: `molecule_id, smiles`

## Machine
MacBook Pro M4 Pro 12CPU / 24GB / 16-core Apple GPU Metal 4. No CUDA - use PyTorch MPS.

## Headless setup
```sh
# Option A
export KAGGLE_TOKEN="KGAT_xxx"
# Option B
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."

opencode mcp list
```

See `docs/kaggle-mcp-no-ui.md` for details. Provide KAGGLE_TOKEN or USERNAME/KEY to activate.

## Deps + one virtual env
One env lives here: `.venv` (CPython 3.13, `uv venv` + `uv sync`).
Declared in `pyproject.toml` + pinned in `uv.lock`: numpy, pandas, pyarrow.
```sh
cd ~/Desktop/enveda-casmi26-molecule-id
uv sync
.venv/bin/python baseline_v0.py  # regenerates submission_v0.csv, no --with flags needed
```
Kaggle CLI is intentionally NOT a project dep — isolated via `uv tool install kaggle`
(`~/.local/bin/kaggle`). Auth: new `~/.kaggle/access_token` (current), not
`kaggle.json` (legacy). Full findings: `docs/kaggle-auth-cli.md`.
