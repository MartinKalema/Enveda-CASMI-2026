"""Isotope agreement features in fork-ranker conventions.
Per candidate: z-scored/log/rank views are computed by the caller over the
molecule's candidate set (mirrors rank_features blocks). Here: raw scores.
- iso_m1: -|log1p(obs M+1) - log1p(expected M+1 from formula)|
- iso_m2: same for M+2 (Br/Cl/S signatures dominate)
- iso_hal: 1 if halogen expectation matches observed M+2 band, else 0
Observed satellites from query spectra (max over molecule's spectra).
Formulas: candidate map supplied by caller (train formula / COCONUT formula).
"""
import numpy as np

_M1 = {"C": 0.0108, "H": 0.000115, "N": 0.00364, "O": 0.00038, "S": 0.0076}
_M2 = {"C": 0.0, "H": 0.0, "N": 0.0, "O": 0.00205, "S": 0.0422,
       "Cl": 0.3199, "Br": 0.9782}


def _parse(formula_str):
    import re
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula_str or ""):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def expected_satellites(formula_str):
    f = _parse(formula_str)
    if not f:
        return None
    m1 = sum(n * _M1.get(e, 0.0) for e, n in f.items())
    m2 = sum(n * _M2.get(e, 0.0) for e, n in f.items())
    return m1, m2


def observed_satellites(spectra, tol=0.01, top_n=40):
    """spectra: iterable of (mzs, intens). Returns (obs_m1, obs_m2)."""
    o1 = o2 = 0.0
    for mzs, intens in spectra:
        mz = np.asarray(mzs, dtype=float)
        it = np.asarray(intens, dtype=float)
        o = np.argsort(-it)[:top_n]
        mz, it = mz[o], it[o]
        r1, r2 = [], []
        for i in range(len(mz)):
            if it[i] <= 0:
                continue
            d1 = np.abs(mz - (mz[i] + 1.003355))
            j = int(np.argmin(d1))
            if d1[j] <= tol:
                r1.append(it[j] / it[i])
            d2 = np.abs(mz - (mz[i] + 2.0057))
            k = int(np.argmin(d2))
            if d2[k] <= tol:
                r2.append(it[k] / it[i])
        if r1:
            o1 = max(o1, float(np.median(r1)))
        if r2:
            o2 = max(o2, float(np.median(r2)))
    return o1, o2


def iso_scores(formula_str, obs1, obs2):
    exp = expected_satellites(formula_str)
    if exp is None:
        return 0.0, 0.0, 0.0
    e1, e2 = exp
    m1 = -abs(np.log1p(obs1) - np.log1p(e1))
    m2 = -abs(np.log1p(obs2) - np.log1p(e2))
    f = _parse(formula_str)
    hal = 1.0 if ((f.get("Br", 0) > 0 and obs2 > 0.5) or
                  (f.get("Cl", 0) > 0 and 0.15 < obs2 < 0.6) or
                  (f.get("Br", 0) == 0 and f.get("Cl", 0) == 0 and obs2 < 0.15)) else 0.0
    return float(m1), float(m2), hal
