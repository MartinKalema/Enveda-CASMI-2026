"""Bout fairness: both sides at full strength, no handicaps."""
import re


def _src():
    return open("/Users/martin/Desktop/enveda-casmi26-molecule-id/v14/stack_bout.py").read()


def test_their_frag_uses_real_scorer():
    s = _src()
    assert "explain_score" in s, "must call their explain_score, not binned overlap"
    assert "np.isin(np.round(m2" not in s, "crude binned overlap still present"


def test_analogs_from_full_library_reps():
    s = _src()
    assert "analog_sim(_libfull" in s, "analogs must use full-library reps"
    assert "_rep, _rep_key, _rep_nm, _rep_ad = L.build_rep(_libfull)" in s


def test_model_dual_view():
    s = _src()
    assert "_merge_peaks" in s or "merged" in s.lower(), "merged view missing"


def test_bout_reports_both_sides():
    s = _src()
    assert 'res["theirs"]' in s and 'res["ours"]' in s
