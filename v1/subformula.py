"""Subformula labelling (MIST-CF lite, RDKit-free).
Assign each MS2 peak a subformula of the candidate precursor formula.
RDBE filter, ppm matching, adduct-adjusted masses.
"""
import numpy as np
from itertools import product

# monoisotopic masses
ELEM_MASS = {
    "C": 12.0, "H": 1.00782503223, "N": 14.00307400443, "O": 15.99491461957,
    "P": 30.9737619985, "S": 31.9720711744, "F": 18.99840316273,
    "Cl": 34.968852682, "Br": 78.9183376, "I": 126.9044719,
    "Na": 22.9897692809, "K": 38.9637074864,
}
ELEM_ORDER = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"]

ADDUCT_DELTA = {
    "[M+H]+": 1.007276, "[M+Na]+": 22.989218, "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823, "[M-H]-": -1.007276, "[M+Cl]-": 34.968853,
    "[M+CH2O2-H]-": 44.998201, "[M+C2H4O2-H]-": 59.013851,
    "[M]+": 0.0, "[M-H2O+H]+": -17.003348, "[M-2H2O+H]+": -35.013913,
    "[2M+H]+": None, "[2M+Na]+": None, "[2M-H]-": None, "[M+2H]2+": None,
}


def parse_formula(s):
    """'C9H8N2O2' -> dict. Handles two-letter elements."""
    import re
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", s):
        if el not in ELEM_MASS:
            return None
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def formula_mass(f):
    return sum(ELEM_MASS[e] * n for e, n in f.items())


def rdbe(f):
    """Ring-double-bond equivalents. None if elements unsupported."""
    c = f.get("C", 0); h = f.get("H", 0); n = f.get("N", 0)
    hal = sum(f.get(e, 0) for e in ("F", "Cl", "Br", "I"))
    for e in f:
        if e not in ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"):
            return None
    return c - (h + hal) / 2 + n / 2 + 1


def enumerate_subformulae(prec_f, max_n=200000):
    """All f ⊆ prec_f with RDBE >= 0, as (counts_tuple, mass). Bounded."""
    keys = [e for e in ELEM_ORDER if e in prec_f]
    counts = [prec_f[e] for e in keys]
    # guard combinatorial explosion (e.g. C30H50...): cap by sampling coarse grid
    total = 1
    for c in counts:
        total *= (c + 1)
    subs = []
    if total <= max_n:
        for combo in product(*[range(c + 1) for c in counts]):
            if all(v == 0 for v in combo):
                continue
            f = dict(zip(keys, combo))
            if (rdbe(f) or -1) < 0:
                continue
            subs.append((combo, sum(ELEM_MASS[e] * n for e, n in zip(keys, combo))))
    else:
        # vectorized random sampling for huge combinatorial spaces
        rng = np.random.default_rng(0)
        k = len(keys)
        cm = np.array(counts)
        draws = rng.integers(0, cm + 1, size=(min(max_n * 3, 600000), k))
        draws = np.unique(draws, axis=0)
        nz = draws[np.any(draws > 0, axis=1)][:max_n]
        idx = {e: i for i, e in enumerate(keys)}
        hal_cols = [idx[e] for e in ("F", "Cl", "Br", "I") if e in idx]
        hal = nz[:, hal_cols].sum(axis=1) if hal_cols else 0
        c = nz[:, idx["C"]] if "C" in idx else 0
        h = nz[:, idx["H"]] if "H" in idx else 0
        n = nz[:, idx["N"]] if "N" in idx else 0
        ok = (c - (h + hal) / 2 + n / 2 + 1) >= 0
        sup = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
        if any(e not in sup for e in keys):
            ok = ok & False
        nz = nz[ok][:max_n]
        mv = np.array([ELEM_MASS[e] for e in keys])
        masses = nz @ mv
        subs = [(tuple(row), float(m)) for row, m in zip(nz.tolist(), masses.tolist())]
    return keys, subs


_SUB_CACHE = {}


def subformula_masses(prec_formula_str, max_n=200000):
    """Cached (keys, masses array) for a precursor formula."""
    hit = _SUB_CACHE.get(prec_formula_str)
    if hit is not None:
        return hit
    prec_f = parse_formula(prec_formula_str)
    if prec_f is None:
        return None
    keys, subs = enumerate_subformulae(prec_f, max_n)
    arr = np.array([m for _, m in subs], dtype=float)
    _SUB_CACHE[prec_formula_str] = (keys, arr)
    return keys, arr


def label_peaks(mzs, intens, prec_formula_str, adduct, ppm=15.0, top_n=20):
    """Greedy: for each top-N peak (by intensity), nearest subformula mass within ppm.
    Returns list of (mz, intensity, subformula_mass or None, ppm_err or None).
    Assumes fragments carry precursor adduct (MIST-CF assumption).
    """
    cached = subformula_masses(prec_formula_str)
    if cached is None:
        return [(m, i, None, None) for m, i in zip(mzs, intens)]
    d = ADDUCT_DELTA.get(adduct)
    if d is None:
        return [(m, i, None, None) for m, i in zip(mzs, intens)]
    mzs = np.asarray(mzs, dtype=float); intens = np.asarray(intens, dtype=float)
    order = np.argsort(-intens)[:top_n]
    _, sub_masses = cached
    out = []
    for idx in order:
        target = mzs[idx] - d  # adduct-adjusted neutral fragment mass
        if len(sub_masses) == 0:
            out.append((mzs[idx], intens[idx], None, None))
            continue
        j = int(np.argmin(np.abs(sub_masses - target)))
        err_ppm = abs(sub_masses[j] - target) / max(target, 1e-9) * 1e6
        if err_ppm <= ppm:
            out.append((mzs[idx], intens[idx], float(sub_masses[j]), float(err_ppm)))
        else:
            out.append((mzs[idx], intens[idx], None, None))
    return out


def explained_intensity(mzs, intens, prec_formula_str, adduct, ppm=15.0, top_n=20):
    """Fraction of top-N intensity explained by subformulae. Core v1 feature."""
    labelled = label_peaks(mzs, intens, prec_formula_str, adduct, ppm, top_n)
    tot = sum(i for _, i, _, _ in labelled)
    exp = sum(i for _, i, m, _ in labelled if m is not None)
    n_hit = sum(1 for _, _, m, _ in labelled if m is not None)
    return (exp / tot if tot > 0 else 0.0), n_hit
