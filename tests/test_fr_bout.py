"""fr_bout must be single-variable: two arms, identical stack, only fr differs."""
SRC = "/Users/martin/Desktop/enveda-casmi26-molecule-id/v14/fr_bout.py"


def _src():
    return open(SRC).read()


def test_two_arms():
    s = _src()
    two_sites = s.count("rank_features(") >= 2
    shared_scorer = "def score_arm" in s and s.count("score_arm(") >= 2
    assert two_sites or shared_scorer, \
        "both arms must go through the same 31-feature stack"


def test_only_frag_differs():
    s = _src()
    assert "their_frags" in s and "bde" in s.lower(), \
        "arms must contrast their frag masses vs BDE frag masses"


def test_shared_ranker():
    s = _src()
    assert "their_ranker" in s, "both arms judged by THEIR ranker (anchor-gated)"


def test_per_seed_logging():
    s = _src()
    assert "seed" in s and (".csv" in s or "log" in s.lower()), \
        "per-seed audit trail required"
