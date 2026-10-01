"""v14 frag-source contract: fr channel uses OUR BDE masses + THEIR scorer.
Fails until submit_v14 swaps FF._frag_masses_wrapper/their_frag_prod.pkl
for frag_up BDE masses (regime-aware) scored with their explain_score."""
import re

SRC = "/Users/martin/Desktop/enveda-casmi26-molecule-id/v14/submit_v14.py"


def _src():
    return open(SRC).read()


def test_bde_mass_source():
    s = _src()
    assert "fragment_masses_bde" in s or "cached_fragments_bde" in s, \
        "fr masses must come from v13.frag_up BDE enumeration"


def test_no_their_frag_masses():
    s = _src()
    assert "_frag_masses_wrapper" not in s, "their unweighted frag masses must go"
    assert "their_frag_prod" not in s, "their frag cache must not feed fr"


def test_regime_aware():
    s = _src()
    assert "_regime" in s or "regime" in s, "BDE charge regime must follow query adduct"


def test_their_scorer_kept():
    s = _src()
    assert "their_explain" in s or "explain_score" in s, \
        "their explain_score stays: ranker was trained on its distribution"
