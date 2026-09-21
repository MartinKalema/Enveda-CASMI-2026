# Kaggle Auth + CLI - What We Discovered (2026-09-21)
Verified headless on MacBook M4 Pro, no UI.

## Correction: new token is current, `kaggle.json` is legacy
- I was wrong to ask for `kaggle.json` username+key. That is now **Option 4 legacy**.
- You were right: the `KGAT_...` token in `~/.kaggle/access_token` is the current method.
- Sources: `Kaggle/kaggle-cli docs/README.md`, `Kaggle/kagglehub README`, `kaggle.com/docs/api`.

## Auth options (new `kaggle` CLI / `kagglehub`, in order tried)
1. OAuth: `kaggle auth login` - browser flow. AVOID for headless.
2. Env: `export KAGGLE_API_TOKEN=KGAT_...` - best for CI/standalone.
3. File: `~/.kaggle/access_token` containing raw token, `chmod 600`. Already set up:
   `mkdir -p ~/.kaggle && echo KGAT_... > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`
   Note: file currently 38 bytes (37-char token + trailing newline). CLI strips it, MCP needs `tr -d '\n\r '`.
4. Legacy: `~/.kaggle/kaggle.json` via Settings > Legacy API Credentials > Create Legacy API Key. NOT needed.

## Verified headless commands (token in file, no env needed except for safety)
```sh
export PATH="$HOME/.local/bin:$PATH"
export KAGGLE_API_TOKEN=$(tr -d '\n\r ' < ~/.kaggle/access_token)
kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra
# -> No submissions found
kaggle competitions leaderboard -c enveda-CASMI26-molecule-id-mass-spectra --show | head
# -> 0.409 Ozymandias31415, 0.388 Randy, 0.387 Udam Liyanage
```
MCP remote also works headless: `POST https://www.kaggle.com/mcp tools/list` with
`Authorization: Bearer $KAGGLE_API_TOKEN` returns 71 tools. OpenCode project
`opencode.jsonc` uses `"oauth": false` + `"Authorization": "Bearer {env:KAGGLE_TOKEN}"`,
so run with `KAGGLE_TOKEN=$(cat ~/.kaggle/access_token) opencode run --standalone ...`
to give the private server the token (shared background service loses env on restart).

## Code-competition submit requirement
`kaggle competitions submit` for this comp needs a kernel, not raw CSV:
`kaggle competitions submit -k <owner/slug> -f submission.csv -m "msg" <competition>`
We have no kernel yet, so `submission_v0.csv` is local-only. Next: wrap
`baseline_v0.py` as notebook, `kaggle kernels push`, then submit.

## Install
- CLI: `uv tool install kaggle` (puts `kaggle` in `~/.local/bin`, isolated from project venv).
- Project env: see `pyproject.toml` + `.venv` in this folder. Never `pip install` system-wide (PEP 668).
