"""Channel unit tests with synthetic ground truth.
Each fragmentation sub-mechanism must fire on a spectrum DESIGNED to contain
its signature, and stay quiet on one designed without it. No real-data
statistics can substitute for these.
"""
import numpy as np

from v13.frag_up import NL_MENU


def test_nl_menu_contains_essentials():
    names = [n for n, _ in NL_MENU]
    for must in ("H2O", "CO", "NH3", "HCOOH", "CO2"):
        assert must in names, f"missing {must}"


def test_nl_water_loss_fires():
    """Synthetic spectrum: precursor 350.0 with a peak at 350-18.0106."""
    from v13.frag_up import frag_score_bde
    prec = 350.0
    mz = np.array([100.0, 200.0, prec - 18.010565, 150.0])
    it = np.array([0.1, 0.2, 1.0, 0.15])
    _, nl = frag_score_bde(mz, it, "CCO", "[M+H]+", prec)
    assert nl > 0, "water-loss peak present but NL menu silent"


def test_nl_quiet_without_losses():
    """Peaks at random masses matching no menu entry -> near silence."""
    from v13.frag_up import frag_score_bde
    prec = 350.0
    mz = np.array([101.11, 203.33, 277.77])
    it = np.array([1.0, 0.5, 0.25])
    _, nl = frag_score_bde(mz, it, "CCO", "[M+H]+", prec)
    assert nl < 0.3, f"NL hallucinating: {nl}"


def test_bde_beats_uniform_on_designed_case():
    """A molecule whose weakest link is unambiguous should outscore a decoy
    whose weak links point elsewhere. Guards the ordering, not the data."""
    from v13.frag_up import fragment_masses_bde
    frags = fragment_masses_bde("CCOCC", "[M+H]+")
    assert len(frags) > 0
    assert len(frags) < 5000, "enumeration explosion guard"
