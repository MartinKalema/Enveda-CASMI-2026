# Lessons learned (read before building v5+)

## Kernel notebook builder (burned us 4 times)
1. `if __name__ == "__main__":` is TRUE in notebooks. Validation blocks
   auto-ran inside the kernel twice. Strip ALL `__main__` guards when
   inlining modules; append exactly one explicit `main()` call.
2. Versioned imports (`from v1...`, `from v2...`, `from v4...`) break in the
   kernel (no packages there). Strip every `^from v\d` / `^import v\d` line;
   modules share one namespace in notebook cells.
3. Hardcoded local paths (`/Users/martin/...`) fail silently as
   FileNotFoundError hours into a run. Builder asserts zero `Users/martin`
   in the final notebook before writing.
4. Patch SOURCES, never generated files. Rebuilding the notebook by
   patching `baseline_v0.py` (unchanged) pushed identical code as v2 AND v3.
   Always `assert` the replaced block exists; fail loud, not silent.
5. Grep-verify before push: no `Users/martin`, no `from v\d`, exactly one
   `main()` call, all cells compile. v10 died on a missed
   `from v2.blend import cosine` because the strip pattern didn't match.

## Kaggle infrastructure
6. Public kernel + private dataset = silently EMPTY `/kaggle/input`
   (no error at push). Keep kernel AND dataset private together.
   (If we ever need public: make the DATASET public first.)
7. Code-competition submit needs all three:
   `kaggle competitions submit -k <owner/slug> -f submission.csv -v N <comp>`.
   Missing `-v` or raw-CSV submit = 400 Bad Request.
8. Kernel failures show in ~1 min (import/path bugs); real runs take 7-20 min.
   RUNNING past 5 min is a good sign. Scoring PENDING 10-60 min is normal queue.
9. `kaggle datasets` has no make-public/make-private: delete + recreate.
   Delete syntax is positional: `kaggle datasets delete <owner/slug> -y`.
10. New CLI auth: `~/.kaggle/access_token` (current). `kaggle.json` is legacy.
    Install via `uv tool install kaggle` (never system pip, PEP 668).

## Modeling (evidence, not opinion)
11. Local disjoint MRR does NOT always transfer: v1 0.099 local → 0.044 LB.
    Cosine 0.088 LB remains the only proven hidden-test signal.
12. Ranker < recall. Hidden NPs aren't in train; biggest lifts come from
    candidate pool (COCONUT 627K new), not scorer tweaks.
13. Test spectra are peak-dense (median 230 vs train 42). Renormalize/truncate
    peaks before any spectral comparison.
14. Shipped `test.parquet` is a train slice: useless for CV, fine for plumbing.
    All validation must be structure-disjoint (inchikey14 minimum).
15. Rare adducts (K+/Cl-/NH4+, ~8 molecules) have 1-5K train spectra each:
    rule-backstop them, don't trust learned scores there.
16. 25-slot economy: ranks 6-25 combined < rank 2. Protect top-5, lottery below.

## Process
17. 5 submissions/day is the scarcest resource. Local validation filters;
    LB confirms. Never spend 2 subs on the same hypothesis.
18. Conventional commits + feature branches; merge to main after submit.
    Generated `submission.csv` at root is untracked (v0's copy kept as reference).
19. Subagents: scoped file outputs, read-only on Kaggle (no submits, no
    visibility changes). Review + commit their files before use.
20. Nothing public without asking. Winners must open-source (MIT) at payout;
    that decision belongs to the human, at that time.
