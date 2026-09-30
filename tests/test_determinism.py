"""Determinism: same data + same params must give identical predictions.
Nondeterministic training was a silent wobble source (no random_state)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def test_gbm_deterministic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 6)), columns=list("abcdef"))
    y = (X["a"] + X["b"] > 0).astype(int)
    ps = []
    for _ in range(2):
        clf = HistGradientBoostingClassifier(random_state=0)
        clf.fit(X, y)
        ps.append(clf.predict_proba(X)[:, 1])
    np.testing.assert_array_equal(ps[0], ps[1])


def test_all_trainers_seeded():
    import re
    for p in ("v11/enrich_train.py", "v13/enrich_bde.py", "v13/gbm_grid.py",
              "v8/rerank25.py", "v5/rerank_gbm.py"):
        try:
            s = open(f"{PROJECT}/{p}").read()
        except FileNotFoundError:
            continue
        for m in re.finditer(r"HistGradientBoostingClassifier\((.*?)\)", s, re.S):
            assert "random_state" in m.group(1), f"unseeded GBM in {p}"


def test_enriched_row_counts_match_input():
    import pandas as pd
    for seed in (10, 11, 12):
        d = pd.read_parquet(
            f"/Users/martin/Desktop/enveda-casmi26-molecule-id/v5/feat_cache/seed{seed}.parquet")
        q = pd.read_parquet(
            f"/Users/martin/Desktop/enveda-casmi26-molecule-id/v13/feat_bde_keyed{seed}.parquet")
        assert len(q) == len(d), f"seed {seed}: {len(q)} != {len(d)}"
        assert set(q.columns) >= {"seed", "truth", "cand", "bde"}
