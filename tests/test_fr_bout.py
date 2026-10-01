"""fr_bout must be single-variable: two arms, identical stack, only fr differs."""
SRC = "/Users/martin/Desktop/enveda-casmi26-molecule-id/v14/fr_bout.py"


def _src():
    return open(SRC).read()


def test_two_arms():
    s = _src()
    factorial = all(k in s for k in ("old+their", "old+bde", "new+their", "new+bde"))
    shared = s.count("rank_features(cfp") == 1
    assert shared and factorial, \
        "2x2 factorial through one shared stack scorer required"


def test_per_query_audit():
    s = _src()
    assert "qid" in s and "truth_lv" in s, \
        "log must carry per-query diagnostics (qid, truth_lv)"


def test_only_frag_differs():
    s = _src()
    assert "their_frags" in s and "bde" in s.lower(), \
        "arms must contrast their frag masses vs BDE frag masses"


def test_shared_ranker():
    s = _src()
    assert "their_ranker" in s, "anchor ranker required"
    assert "ranker_bde" in s, "retrained BDE ranker must be judged too"


def test_per_seed_logging():
    s = _src()
    assert "seed" in s and (".csv" in s or "log" in s.lower()), \
        "per-seed audit trail required"
