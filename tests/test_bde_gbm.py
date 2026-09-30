"""TDD for BDE-ladder GBM enrichment: synthetic ground truth first."""
import numpy as np

from v13.enrich_bde import ladder_score


def test_ladder_fires_on_designed_hit():
    """Fragment at 200.0 + proton, peak exactly there -> must score ~1."""
    frags = np.array([150.0, 200.0, 250.0])
    mz = np.array([201.0073, 99.0])
    it = np.array([1.0, 0.01])
    assert ladder_score(mz, it, frags) > 0.9


def test_ladder_quiet_on_miss():
    frags = np.array([150.0, 200.0, 250.0])
    mz = np.array([111.11, 222.22])
    it = np.array([1.0, 0.5])
    assert ladder_score(mz, it, frags) == 0.0


def test_ladder_empty_inputs():
    assert ladder_score([], [], np.array([1.0])) == 0.0
    assert ladder_score([1.0], [1.0], np.array([])) == 0.0


def test_ladder_h_shift_matters():
    """A peak matching ONLY via H-2 shift must score with ladder, zero without.
    Guards the exact superiority under test (ladder vs single-+H)."""
    frags = np.array([200.0])
    mz = np.array([200.0 + 1.007276 - 2 * 1.007825 + 1.007276])
    it = np.array([1.0])
    assert ladder_score(mz, it, frags) > 0.9
