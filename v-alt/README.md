# v-alt: formula-first isomer retrieval (NO cosine floor)

Structurally different evidence: no spectrum<->spectrum similarity anywhere
(no cosine, entropy, Tanimoto, fingerprints, library matching).

- `alt_score.py` — scorers: `fast_explained` (spectrum vs formula subformulae),
  `frag_fit` (structure->spectrum forward fragmentation fit, RDKit 1-2 bond
  cleavages), `loss_bonus` (peaks vs common neutral-loss constants),
  `local_cosine` (validation anchor ONLY, never used for ranking).
- `validate_alt.py` — structure-disjoint (inchikey14) validation, 40 queries x
  3 seeds (n=120), alt variants + same-split cosine anchor.
  Run: `.venv/bin/python v-alt/validate_alt.py`
- `submit_alt.py` — production pipeline writing `submission.csv`
  (mass window -> formula funnel -> frag-fit isomer rank; `ALT_COCONUT=1`
  to add COCONUT candidates).
- `frag_disk.pkl` — persistent RDKit fragment-mass cache (built on first run).

See `docs/alt-paradigm.md` for design, numbers, and verdict.
