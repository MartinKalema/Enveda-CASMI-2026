"""Reusable spectrum cleaning: adduct correction, quality filter, denoising.

All thresholds come from the v-clean audits (see docs/data-cleaning.md).
Pure functions work row-wise; batch helpers work on DataFrames.
"""
import os

import numpy as np
import pandas as pd

from adducts import adduct_shift, expected_mz, formula_mass, within_tol
from audit_peaks import cosine, denoise

PPM = 10.0
ABS_DA = 0.01
ERR_PPM_CUT = 20.0
DENOISE_THR = 0.01
DENOISE_TOPN = 200

_CAND_CACHE = {}

# precursor_error_ppm > cut means "mislabeled" ONLY in these libs
# (audit: P(adduct-mismatch | err>20ppm) >= 0.86; pluskal 0.0002, riken 0.61,
# enveda ~0 -> the column follows different references there).
ERR_TRUSTED_LIBS = frozenset(
    ['gnps', 'massbank', 'mona', 'msdial', 'drug_plus', 'spectraverse'])


def candidates(polarity):
    """Adduct candidates for best-fit search, restricted to one polarity."""
    if polarity in _CAND_CACHE:
        return _CAND_CACHE[polarity]
    cands = ['[M+H]+', '[M+Na]+', '[M+K]+', '[M+NH4]+', '[M-H2O+H]+',
             '[M-2H2O+H]+', '[M]+', '[M+2H]2+', '[2M+H]+', '[2M+Na]+',
             '[M-H]-', '[M+Cl]-', '[M+CH2O2-H]-', '[M+C2H4O2-H]-',
             '[2M-H]-', '[M-CH3]-', '[M-2H]-']
    pos = [a for a in cands if a.endswith('+') or a.endswith('2+')]
    neg = [a for a in cands if a.endswith('-')]
    _CAND_CACHE['+'] = pos
    _CAND_CACHE['-'] = neg
    return _CAND_CACHE.get(polarity, cands)


def check_label(precursor_mz, molecular_formula, adduct):
    """True/False/None (None = not auditable: bad formula or adduct)."""
    m = formula_mass(molecular_formula)
    e = expected_mz(m, adduct)
    if m is None or e is None or precursor_mz is None:
        return None
    try:
        if np.isnan(precursor_mz):
            return None
    except TypeError:
        return None
    return within_tol(precursor_mz - e, e, PPM, ABS_DA)


def correct_adduct(precursor_mz, molecular_formula, polarity='+'):
    """Best-fit adduct for this precursor+formula, or None if none fits."""
    m = formula_mass(molecular_formula)
    if m is None or precursor_mz is None:
        return None
    best, best_r = None, None
    for a in candidates(polarity):
        e = expected_mz(m, a)
        if e is None:
            continue
        r = abs(precursor_mz - e)
        if best_r is None or r < best_r:
            best, best_r = a, r
    if best is None:
        return None
    e = expected_mz(m, best)
    return best if within_tol(best_r, e, PPM, ABS_DA) else None


def neutral_mass(precursor_mz, adduct):
    """Precursor-derived neutral mass (NaN if adduct unknown)."""
    # generic inverse: M = (z*mz - shift)/m  (exact for all parsed adducts)
    p = adduct_shift(adduct)
    if p is None:
        return np.nan
    m, shift, z = p
    return (z * precursor_mz - shift) / m


def clean_spectrum(mzs, ints, thr=DENOISE_THR, topn=DENOISE_TOPN):
    """Denoise one spectrum: intensity floor + top-N cap, mz-sorted."""
    return denoise(mzs, ints, thr, topn)


def flag_frame(df):
    """Per-row quality flags: label_ok, corrected_adduct, drop.

    Expects columns: precursor_mz, molecular_formula, adduct,
    ionization_mode, precursor_error_ppm.
    """
    out = pd.DataFrame(index=df.index)
    out['label_ok'] = [check_label(p, f, a) for p, f, a in
                       zip(df['precursor_mz'], df['molecular_formula'], df['adduct'])]
    need_fix = out['label_ok'] == False  # noqa: E712
    corr = pd.Series([None] * len(df), index=df.index, dtype='object')
    sub = df[need_fix]
    pol = (sub['ionization_mode'] == 'negative').map({True: '-', False: '+'})
    for idx, (p, f, po) in zip(sub.index,
                               zip(sub['precursor_mz'], sub['molecular_formula'], pol)):
        corr.loc[idx] = correct_adduct(p, f, po)
    out['corrected_adduct'] = corr
    out['rescued'] = out['corrected_adduct'].notna()
    big_err = (df['precursor_error_ppm'].abs() > ERR_PPM_CUT).fillna(False)
    in_trusted = df['ingest_lib'].isin(ERR_TRUSTED_LIBS) if 'ingest_lib' in df else True
    out['drop_20ppm'] = (big_err & in_trusted & ~out['rescued'])
    out['drop_adduct'] = need_fix & ~out['rescued']
    out['drop'] = out['drop_20ppm'] | out['drop_adduct']
    return out


def clean_train(df):
    """Attach flags + corrected adduct + denoised spectra. Returns new frame."""
    flags = flag_frame(df)
    out = pd.concat([df.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
    out['adduct_used'] = np.where(out['rescued'], out['corrected_adduct'], out['adduct'])
    out['neutral_used'] = [neutral_mass(p, a) for p, a in
                           zip(out['precursor_mz'], out['adduct_used'])]
    dn = [clean_spectrum(m, v) for m, v in
          zip(df['ms2_mzs'], df['ms2_normalized_intensities'])]
    out['dn_mzs'] = [m for m, _ in dn]
    out['dn_ints'] = [v for _, v in dn]
    out['dn_n'] = [len(m) for m, _ in dn]
    return out
