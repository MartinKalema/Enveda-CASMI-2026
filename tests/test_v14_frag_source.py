"""v14 production contract: the SHIPPED configuration.
Floor top-5 (our cosine) + THEIR frag masses + THEIR scorer + THEIR ranker,
with the SMILES key-space fix (lib_sim/build_rep keyed by SMILES).
Class-2 bouts killed the BDE mass-source (non-lever, +-0.003) and the
retrain (-0.065): this file pins the validated combination.
"""
import re

SRC = "/Users/martin/Desktop/enveda-casmi26-molecule-id/v14/submit_v14.py"


def _src():
    return open(SRC).read()


def test_their_frag_masses():
    s = _src()
    assert "their_frag_prod" in s, "fills use THEIR frag masses"
    assert "_frag_masses_wrapper" not in s


def test_no_bde_in_production():
    s = _src()
    assert "bde_frag_" not in s and "bde_regime" not in s and "fragment_masses_bde" not in s, \
        "BDE killed by class-2 bout: must not feed production"


def test_their_scorer_and_ranker():
    s = _src()
    assert "their_explain" in s or "explain_score" in s
    assert "their_ranker.pkl" in s, "production judged by THEIR ranker"


def test_smiles_keyspace():
    s = _src()
    assert re.search(r"lib_hits\.get\(s", s), \
        "lib lookup keyed by candidate SMILES (matches fixed lib_sim)"
    assert "L.build_rep" in s and "L.analog_sim" in s and "L.lib_sim" in s


def test_floor_top5_pinned():
    s = _src()
    assert "top5" in s and "seen" in s, "floor top-5 pinned ahead of ranker fills"
