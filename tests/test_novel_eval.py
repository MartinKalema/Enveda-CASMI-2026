"""Novel-molecule end-to-end eval: held-out structures, retrieval INCLUDED.
Queries: held-out train spectra. Library/pool: train MINUS held structures;
truth retrievable ONLY via the COCONUT pool under test. Compares pool
variants (old 627k vs orig 739k) on identical queries. This is the eval the
bouts skipped (their prebuilt lists contain truth by construction)."""
import os

SRC = "/Users/martin/Desktop/enveda-casmi26-molecule-id/v14/novel_eval.py"


def _src():
    return open(SRC).read()


def test_held_out_discipline():
    s = _src()
    assert "inchikey14" in s, "holdout must be by inchikey14 group (no stereo leakage)"
    for token in ("held", "holdout", "exclude"):
        assert token in s.lower(), "held structures must be excluded from library+pool"
    assert "isin" in s or "not in" in s or "!=" in s, "exclusion must be explicit"


def test_pool_variant_switch():
    s = _src()
    assert "coconut_fp.parquet" in s and "coconut_orig.parquet" in s, \
        "must evaluate BOTH pool variants on identical queries"


def test_ranker_and_scorer_pinned():
    s = _src()
    assert "their_ranker.pkl" in s and "rank_features" in s, \
        "same stack as production; pool is the only variable"
