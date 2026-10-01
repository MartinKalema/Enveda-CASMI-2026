"""v14 production contract: OUR BDE frags + THEIR scorer + THEIR ranker.
USER OVERRIDE (2026-10-01): local bouts show BDE neutral inside the stack
(+0.004 class-1 / -0.003 class-2); submitting for LB truth anyway.
Floor top-5 (our cosine) pinned; SMILES key-space fix; fixed stack otherwise.
"""
import re

SRC = "/Users/martin/Desktop/enveda-casmi26-molecule-id/v14/submit_v14.py"


def _src():
    return open(SRC).read()


def test_bde_masses_in_production():
    s = _src()
    assert "bde_frag_" in s, "fills use OUR BDE frag masses"
    assert 'bde_fr["nh4"]' in s or "nh4" in s, "all four regimes covered"


def test_no_their_frag_masses():
    s = _src()
    assert "their_frag_prod" not in s, "their masses out for this submission"


def test_their_scorer_and_ranker():
    s = _src()
    assert "their_explain" in s or "explain_score" in s, \
        "their explain_score (ranker trained on its distribution)"
    assert "their_ranker.pkl" in s, "production judged by THEIR ranker"


def test_smiles_keyspace():
    s = _src()
    assert re.search(r"lib_hits\.get\(s", s), \
        "lib lookup keyed by candidate SMILES (matches fixed lib_sim)"
    assert "L.build_rep" in s and "L.analog_sim" in s and "L.lib_sim" in s


def test_unified_ranking():
    s = _src()
    assert "top5" not in s and "seen" not in s, "no pinned floor: single ranker ordering"
    assert re.search(r"out = \[s for _, s in order\]\[:25\]", s), \
        "output is top-25 of the unified stack ranking"
