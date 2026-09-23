"""v-alt scoring: formula-first isomer retrieval, ZERO spectrum<->spectrum similarity.

Paradigm (survey rank 2, MIST-CF style):
  Stage 1 (formula ranking): rank candidate formulae by subformula explained-intensity
    (query spectrum vs formula physics; sorted-search matching, no cosine anywhere).
  Stage 2 (isomer ranking): rank same-formula isomers by forward fragmentation fit
    (RDKit 1-2 bond cleavage fragment masses vs query peaks; structure->spectrum).
  Tiebreak: neutral-loss bonus (query peaks vs common-loss mass list; spectrum-vs-constant).

Nothing here ever compares two spectra. No cosine, no entropy, no Tanimoto, no fingerprints.
"""
import numpy as np
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _os, _sys

from v1.subformula import (
    ADDUCT_DELTA, parse_formula, subformula_masses, ELEM_MASS,
)
from v7.metfrag import fragment_masses

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"

_SORTED_SUB_CACHE = {}


def sorted_sub_masses(formula_str):
    hit = _SORTED_SUB_CACHE.get(formula_str)
    if hit is None:
        cached = subformula_masses(formula_str)
        if cached is None:
            return None
        _, arr = cached
        hit = np.sort(np.asarray(arr, dtype=float))
        _SORTED_SUB_CACHE[formula_str] = hit
    return hit


def neutral_mass(prec, adduct):
    if adduct == "[2M+H]+":
        return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+":
        return (prec - 22.989218) / 2
    if adduct == "[2M-H]-":
        return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def fast_explained(mzs, intens, formula_str, adduct, ppm=15.0, top_n=20):
    """Fraction of top-N peak intensity explainable as subformulae of formula_str.

    Vectorized searchsorted matching; mathematically ~= v1.explained_intensity
    (nearest-subformula within ppm) but O(P log S) instead of O(P*S).
    Pure spectrum-vs-formula physics.
    """
    subs = sorted_sub_masses(formula_str)
    if subs is None or len(subs) == 0:
        return 0.0
    d = ADDUCT_DELTA.get(adduct)
    if d is None:
        return 0.0
    mzs = np.asarray(mzs, dtype=float)
    intens = np.asarray(intens, dtype=float)
    if len(mzs) == 0:
        return 0.0
    o = np.argsort(-intens)[:top_n]
    pmz, pit = mzs[o], intens[o]
    tot = pit.sum()
    if tot <= 0:
        return 0.0
    targets = pmz - d
    idx = np.searchsorted(subs, targets)
    hit = np.zeros(len(targets), dtype=bool)
    for jj in (0, -1):
        j = np.clip(idx + jj, 0, len(subs) - 1)
        err = np.abs(subs[j] - targets) / np.maximum(targets, 1e-9) * 1e6
        hit |= err <= ppm
    return float((pit[hit]).sum() / tot)


# Common neutral losses (monoisotopic Da), spectrum-vs-constant evidence.
COMMON_LOSSES = np.array(sorted([
    18.010565,   # H2O
    17.026549,   # NH3
    27.994915,   # CO
    43.98983,    # CO2
    27.010899,   # HCN
    46.005478,   # HCOOH / CH2O2
    60.021129,   # C2H4O2 (acetic acid)
    15.994915,   # O (oxidation delta)
    32.026215,   # CH3OH
    46.041865,   # C2H6O (ethanol)
    162.052824,  # hexose C6H10O5
    146.057909,  # deoxyhexose C6H10O4
    132.042259,  # pentose C5H8O4
    176.032088,  # glucuronic C6H8O6
    79.956817,   # SO3
    97.967379,   # H2SO4
    14.01565,    # CH2
    28.0313,     # C2H4
    30.010899,   # CH2O
    44.026215,   # C2H4O
    42.010565,   # C2H2O (ketene)
    56.026215,   # C3H4O
    2.01565,     # H2
    1.007825,    # H radical
]))
FRAG_AD = {
    "[M+H]+": 1.007276, "[M+Na]+": 22.989218, "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823, "[M-H]-": -1.007276, "[M+Cl]-": 34.968853,
    "[M+CH2O2-H]-": 44.998201, "[M+C2H4O2-H]-": 59.013851, "[M]+": 0.0,
}


def loss_bonus(mzs, intens, qneutral, adduct, tol=0.01, top_n=60):
    """sqrt-intensity fraction of peaks matching qneutral - common_loss.

    qneutral comes from the query precursor; losses are constants.
    No second spectrum involved.
    """
    if not np.isfinite(qneutral):
        return 0.0
    d = FRAG_AD.get(adduct, 1.007276)
    mzs = np.asarray(mzs, dtype=float)
    intens = np.asarray(intens, dtype=float)
    if len(mzs) == 0:
        return 0.0
    o = np.argsort(-intens)[:top_n]
    pmz, pit = mzs[o], intens[o]
    w = np.sqrt(np.maximum(pit, 0))
    tot = w.sum()
    if tot <= 0:
        return 0.0
    pneu = pmz - d
    losses = qneutral - pneu
    idx = np.searchsorted(COMMON_LOSSES, losses)
    hit = np.zeros(len(losses), dtype=bool)
    for jj in (0, -1):
        j = np.clip(idx + jj, 0, len(COMMON_LOSSES) - 1)
        hit |= np.abs(COMMON_LOSSES[j] - losses) <= tol
    hit &= losses > 0
    return float(w[hit].sum() / tot)


_FRAG_DISK = {}
_FRAG_PATH = f"{PROJECT}/v-alt/frag_disk.pkl"


def load_frag_disk():
    import os
    import pickle
    global _FRAG_DISK
    if os.path.exists(_FRAG_PATH):
        try:
            with open(_FRAG_PATH, "rb") as f:
                _FRAG_DISK = pickle.load(f)
        except Exception:
            _FRAG_DISK = {}
    return _FRAG_DISK


def save_frag_disk():
    import pickle
    with open(_FRAG_PATH, "wb") as f:
        pickle.dump(_FRAG_DISK, f, protocol=4)


def frag_masses_cached(smiles):
    hit = _FRAG_DISK.get(smiles)
    if hit is None:
        hit = np.array(sorted(fragment_masses(smiles, max_bonds=2)), dtype=float)
        _FRAG_DISK[smiles] = hit
    return hit


def frag_fit(mzs, intens, smiles, adduct, tol=0.01, top_n=100):
    """sqrt-intensity fraction of query peaks explained by structure fragments.

    Forward simulation (structure->spectrum); query spectrum is the only spectrum.
    """
    frags = frag_masses_cached(smiles)
    if len(frags) == 0:
        return 0.0
    d = FRAG_AD.get(adduct, 1.007276)
    mzs = np.asarray(mzs, dtype=float)
    intens = np.asarray(intens, dtype=float)
    o = np.argsort(-intens)[:top_n]
    pmz, pit = mzs[o], intens[o]
    w = np.sqrt(np.maximum(pit, 0))
    tot = w.sum()
    if tot <= 0:
        return 0.0
    targets = pmz - d
    idx = np.searchsorted(frags, targets)
    hit = np.zeros(len(targets), dtype=bool)
    for jj in (0, -1):
        j = np.clip(idx + jj, 0, len(frags) - 1)
        hit |= np.abs(frags[j] - targets) <= tol
    return float(w[hit].sum() / tot)


def local_cosine(mz1, it1, mz2, it2, tol=0.02):
    """Anchor ONLY (v0 cosine floor). Never used by the alt pipeline."""
    a = np.asarray(it1, dtype=float)
    b = np.asarray(it2, dtype=float)
    na = float(np.sqrt((a * a).sum()))
    nb = float(np.sqrt((b * b).sum()))
    if na == 0 or nb == 0:
        return 0.0
    m1 = np.asarray(mz1, dtype=float)
    m2 = np.asarray(mz2, dtype=float)
    o1 = np.argsort(m1)
    o2 = np.argsort(m2)
    m1, a = m1[o1], a[o1]
    m2, b = m2[o2], b[o2]
    i = j = 0
    num = 0.0
    while i < len(m1) and j < len(m2):
        dd = m1[i] - m2[j]
        if abs(dd) <= tol:
            num += a[i] * b[j]
            i += 1
            j += 1
        elif dd < 0:
            i += 1
        else:
            j += 1
    return num / (na * nb)
