# Kaggle MCP - No UI Interaction Research
Project: Enveda CASMI 2026 - Molecule ID From Mass Spectra
Date: 2026-09-21
Machine: MacBook Pro Apple M4 Pro, 12 CPU, 24GB RAM, 16-core Apple GPU (Metal 4)

## Goal
Interact with Kaggle fully headless from OpenCode (no browser OAuth click, no `/mcps` UI).

## Finding: 2 viable headless paths

### Option A - Official Kaggle Remote MCP (recommended for metadata)
- URL: `https://www.kaggle.com/mcp`
- Docs: `https://www.kaggle.com/docs/mcp`
- Auth headless: Token Authentication, NOT OAuth.
  - Get token: Kaggle > Settings > Generate New Token > Copy. Starts with `KGAT`.
  - Header: `Authorization: Bearer KGAT_xxx`
- OpenCode V2 config (no UI because `oauth: false`):
```jsonc
{
  "mcp": {
    "servers": {
      "kaggle-official": {
        "type": "remote",
        "url": "https://www.kaggle.com/mcp",
        "oauth": false,
        "headers": { "Authorization": "Bearer {env:KAGGLE_TOKEN}" }
      }
    }
  }
}
```
- CLI equivalent:
```sh
opencode mcp add kaggle-official --url https://www.kaggle.com/mcp --header "Authorization=Bearer {env:KAGGLE_TOKEN}"
# then edit opencode.jsonc to set "oauth": false, otherwise OpenCode tries OAuth discovery
```
- Tools provided: Notebooks (manage/start/cancel), Competitions (search/metadata/download/submit), Datasets (list files/metadata/search), Models (create/get/download/list), Benchmarking (CreateBenchmarkTaskFromPrompt, GetBenchmarkLeaderboard).
- Verify headless:
```sh
export KAGGLE_TOKEN="KGAT_xxx"
opencode mcp list
# want: ✓ kaggle-official connected (not needs authentication)
opencode run "via kaggle-official list tools, then get metadata for enveda-CASMI26-molecule-id-mass-spectra"
```

### Option B - Community Local MCP `mcp-server-kaggle` (best for competitions download/submit)
- PyPI: `mcp-server-kaggle 0.1.1`, Source: `https://github.com/Seif-Sameh/Kaggle-mcp`
- Requires: Python >=3.10, `uv`, classic Kaggle API creds (`KAGGLE_USERNAME` + `KAGGLE_KEY` from `kaggle.json`)
- No OAuth, pure env vars = fully headless.
- OpenCode V2 config already in `./opencode.jsonc`:
```jsonc
"kaggle-local": {
  "type": "local",
  "command": ["uvx", "mcp-server-kaggle"],
  "environment": {
    "KAGGLE_USERNAME": "{env:KAGGLE_USERNAME}",
    "KAGGLE_KEY": "{env:KAGGLE_KEY}"
  }
}
```
- CLI equivalent:
```sh
export KAGGLE_USERNAME="yourname"
export KAGGLE_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
opencode mcp add kaggle-local --env KAGGLE_USERNAME=$KAGGLE_USERNAME --env KAGGLE_KEY=$KAGGLE_KEY -- -- uvx mcp-server-kaggle
```
- Tools: Competitions (8 tools: list/download/submit/leaderboard), Datasets (search/download/create), Kernels (list/push/pull), Models (14 tools).
- Test headless:
```sh
uvx mcp-server-kaggle --help
KAGGLE_USERNAME=... KAGGLE_KEY=... opencode run "list Kaggle competitions via kaggle-local"
```

## What NOT to do for no-UI
- Do NOT leave `"oauth"` unset on remote server without token - OpenCode will show `needs authentication` and require `/mcps` browser flow.
- Do NOT run `opencode mcp auth kaggle-official` - that starts interactive OAuth.
- Do NOT hardcode secrets in `opencode.jsonc`. Use `{env:VAR}` only.

## For this challenge
Competition slug (to confirm via MCP): likely `enveda-casmi26-molecule-id-mass-spectra`
Local data already copied to `./data/` (train 2.5M rows, test 1213 rows). Use MCP for:
1. `competitions metadata / leaderboard` check
2. `competitions download` if refresh needed (you already have files)
3. `competitions submit` submission.csv headless

## Needed from you
- For Option A: `KAGGLE_TOKEN` (KGAT_...)
- For Option B: `KAGGLE_USERNAME` + `KAGGLE_KEY` (contents of kaggle.json)
Provide either pair, I will `export` and verify with `opencode mcp list` + `opencode run --format json` with zero UI.

## PC specs note (ML/GPU)
- `arm64 M4 Pro, 12 cores, 24GB unified RAM, 16-core Apple GPU Metal 4`
- No NVIDIA/CUDA. On macOS use:
  - `torch` with `mps` backend (`torch.backends.mps.is_available()`)
  - `tensorflow-metal` plugin, or sklearn/lightgbm CPU
  - 2.8GB train.parquet (2.5M spectra) fits in 24GB RAM with pyarrow/pandas chunking
- Next step: `uv init` + `uv add pandas pyarrow torch scikit-learn` and test MPS.
