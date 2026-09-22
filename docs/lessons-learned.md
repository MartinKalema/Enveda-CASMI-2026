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

## Validation discipline (v4 post-mortem, LB 0.013)
21. n=30 single-seed disjoint validation is UNRELIABLE. Same cosine channel
    scored MRR 0.099 (seed 0, v1 split) vs 0.002 (seed 3, v4 split) - 50x
    swing from split noise alone. v4's "analog 30x cosine" was relative order
    on an uncalibrated split and did not transfer (LB 0.013).
22. Always include a known-behavior anchor (v0-cosine) in every validation run.
    If the anchor's score differs >2x from its LB (0.088), the split is
    unrepresentative - distrust all relative orderings from it.
23. Validate with n>=100 across >=3 seeds before spending a submission.
    One split, one seed, n=30 is how v4 burned a sub for 0.013.
24. New channels replace the backbone ONLY after beating the anchor on a
    calibrated split. Entropy never beat cosine on a calibrated split;
    it replaced it on an uncalibrated one.

## Hybrid discipline (v5 calibration, 3 seeds x 50)
25. Disjoint protocol measures NOVEL-structure skill only - and hidden test is
    MIXED knowns + novels (v0/v3 prove knowns score). Local ~0.00 does not
    contradict LB 0.088; they measure different subpopulations. Never infer
    "channel X is worthless" from disjoint zeros alone.
26. Never replace a proven floor channel. v4 swapped v0-cosine top-5 for
    entropy and fell 0.095 -> 0.013. New channels may only ADD slots below
    the proven top-5, never take them. (v5 design: cosine 1-5, analog 6-25.)
27. Seed-0 v1 split (MRR 0.099) vs seeds 3/10/11/12 (~0.000): same protocol,
    50x swing. n=30-50 single seed cannot rank channels. n>=100 x >=3 seeds,
    anchor-gated, or it didn't happen.

## Evidence over authority (v7 floor validation)
28. Entropy similarity is DEAD on our data, three strikes: retrieval-style
    0.527 vs cosine 0.970 (truth pooled), disjoint weak, v4 LB 0.013 after
    replacing cosine with it. The survey's "entropy > cosine" does not
    transfer (different cleaning/intensity powers). Drop it; stop retrying.
29. Pooled-truth retrieval saturates (~0.97): near-duplicate spectra always
    match. Useful ONLY for channel comparison (cosine >> entropy), never as
    an absolute predictor. And a disjoint run that leaves truth spectra in
    the pool scores 1.000 - always exclude held groups from BOTH candidates
    AND scoring spectra (v7 first disjoint run leaked exactly this way).
