"""v7 floor: formula-mass index, tight windows, entropy-weighted cosine."""
import numpy as np
from v1.subformula import ELEM_MASS, parse_formula

ADDUCT_DELTA = {
    "[M+H]+": 1.007276, "[M+Na]+": 22.989218, "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823, "[M-H]-": -1.007276, "[M+Cl]-": 34.968853,
    "[M+CH2O2-H]-": 44.998201, "[M+C2H4O2-H]-": 59.013851,
    "[M]+": 0.0, "[M-H2O+H]+": -17.003348, "[M-2H2O+H]+": -35.013913,
}


def formula_exact_mass(fstr):
    from v1.subformula import parse_formula as pf
    f = pf(fstr)
    if f is None: return None
    try:
        return sum(ELEM_MASS[e] * n for e, n in f.items())
    except KeyError:
        return None


def neutral_from_precursor(prec, adduct):
    if adduct == "[2M+H]+": return (prec - 1.007276) / 2
    if adduct == "[2M+Na]+": return (prec - 22.989218) / 2
    if adduct == "[2M-H]-": return (prec + 1.007276) / 2
    d = ADDUCT_DELTA.get(adduct)
    return prec - d if d is not None else np.nan


def hybrid_tol(mass, ppm=10.0, abs_da=0.01):
    return max(mass * ppm / 1e6, abs_da)
