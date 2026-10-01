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

## Tooling discipline
30. Never pipe validation output through `tail`: it silently eats seeds.
    Seed 31's result was lost to `| tail -n 4`; rerun cost 20 minutes.
31. `python -c "import pkg.mod"` executes the module: `if __name__ ==
    "__main__"` guards fire and validation runs TWICE (default seed + yours).
    Import the function only (`from pkg.mod import run`), or guard with
    `if __name__ == "__main__"` awareness.

## Tempo discipline (wasted 2026-09-22)
32. Never idle on a PENDING score. Submissions score on Kaggle's clock while
    builds run on ours - the two proceed in parallel or a day dies. Always
    have the next version training/building while the last one queues.
    Fills plateaued 0.093-0.095 across FOUR versions (v3/v5/v6/v7) before
    anyone admitted the floor must change.

## Negative results that stay dead
33. BDE-ordered cleavage + neutral-loss ladder: disjoint MRR 0.279 vs plain
    1-2-bond 0.272 (n=30, same split) - no signal. Do NOT port to fork.
    Fragmentation physics beyond BFS buys nothing at ranking time.

## Ablation discipline (BDE vs NL menu)
34. NL menu ablation: WITH-NL 0.279 vs WITHOUT-NL 0.279, identical rankings.
    The 0.011 NL signal never flips a single rank - dead weight. BDE ordering
    alone carries the whole win over their channel. Lesson: ablate every
    bolt-on; a positive-looking signal (0.011 vs 0.000) can still be
    ranking-irrelevant.

## Port verification (the +H incident)
35. A hand-port is guilty until proven innocent: ours produced 47 fragments
    per molecule vs 94 locally (exactly half, ~1 Da shift) because the +H
    rearrangement variants were dropped in transcription. Caught only by a
    differential test comparing ported vs local outputs on real inputs.
36. Differential tests are now mandatory for every port: same inputs must
    give identical outputs before the code ships anywhere near a submission.
    tests/test_port_parity.py is the template (bond selection + end-to-end
    scores + leak guards).
37. Never submit without confirming new code actually ran: check kernel
    version output freshness, not just COMPLETE status. A COMPLETE badge on
    a stale version burned a submission for an identical 0.317.

## Harness bugs produce false findings (frag channel incident)
38. A stray pasted decorator made their code unrunnable in my harness; it
    scored all zeros and I reported "their channel is dead." Corrected
    numbers: OLD mean 0.247 (94% nonzero), rank-corr 0.531 with ours.
    Rule: a surprising zero is a harness suspect first, a finding second -
    verify the harness runs the target code correctly before believing any
    number it prints, especially zeros.

## H-shift ladder isolation (local, n=30)
39. Same fragments, same peaks, only H-ladder varies: 5-state ladder MRR
    0.482 vs single-+H 0.224. The ladder more than doubles fragmentation
    scoring. It is the single biggest verified sub-component win: BDE
    enumeration selects better pieces, the ladder matches much more of them.

## Hyperparameter grid (GBM, 3-fold seed holdout, 20 configs)
40. Winner: lr 0.05, leaves 31, l2 1.0 (mean 0.084). Mine-made-up 0.073,
    theirs 0.056. Tuning is worth ~+0.01 over habits on 3 features; the
    8-feature set was worth +0.10. Features dominate params 10-to-1.
    Adopted as default for all future ranker training.

## Params do not transfer across feature sets (GBM final)
41. Grid-winning params (lr 0.05/31 leaves/l2 1.0, tuned on 3 feats) scored
    0.203/0.134/0.133 (mean 0.157) on 8 feats vs 0.202/0.165/0.177
    (mean 0.181) with the old params. The "adopt as default" conclusion was
    overgeneralized: tune per feature set, never port params. v11/gbm_full.pkl
    stays the production ranker; v13/gbm_final.pkl is NOT shipped.

## TDD is the process (rule, not aspiration)
42. No measurement counts without a green test suite behind the harness that
    produced it. Tests are written FIRST (watch them fail), implementation
    second, numbers last. Background runs without tests are how the +H bug,
    the decorator bug, the tail-pipe losses, and the positional merge all
    shipped. The suite lives in tests/ and CI is `pytest tests/ -q`.

## Stack-vs-stack bout (n=150, identical queries)
43. Theirs 0.595 vs ours 0.445. Component wins do not survive integration:
    our BDE beats their frag alone, but their 31-feature ranker combining
    four channels beats our 9-feature GBM combining six. Breadth +
    calibration beats best-single-channel. Next: our channels as features
    inside THEIR ranker shape (retrain), not our ranker with their pieces.

## v14 kernel build (v32 COMPLETE)
44. Kernel-inlined code has different name resolution than module code:
    F./L. aliases vanish, and bare `neutral_mass` collides (vectorized lib
    version vs scalar run version). Fix: alias `lib_neutral_mass` at end of
    lbs cell, map only that name. Static test tests/test_kernel_static.py
    (module-prefix + run-global resolution) plus file-backed exec smoke
    (numba cache=True needs real files, fails under bare exec()) catch all
    three failure classes locally before a Kaggle round trip.
45. Kaggle CLI submit returning 400 even for sample_submission.csv =>
    systemic (token scope/endpoint), not file format. Validate format
    locally (400 rows, ids match sample, 25/row) and submit via browser.

## fr bout: BDE masses under their stack (n=150, seeds 10/11/12)
46. BDE-fr 0.824 vs their-fr 0.820 (+0.004): NO-GO, gate |Δ|>=0.012 not met.
    Per-seed +0.014/+0.008/-0.012 — direction flips by query mix, i.e. noise.
    Frag is 4/31 features and the ranker was trained on explain_score over
    UNWEIGHTED masses; swapping the mass source without retraining moves
    features off-distribution with no consistent gain. Lesson: mass-source
    swaps need ranker retraining to count, not drop-in precompute.
