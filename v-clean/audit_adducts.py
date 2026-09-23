"""Adduct-label audit: precursor-derived neutral mass vs formula exact mass.

Usage: .venv/bin/python v-clean/audit_adducts.py
Reads data/train.parquet (READ-ONLY), writes data-clean/adduct_audit.parquet
(per-row residuals + best-fit adduct) and prints mislabel rates per ingest_lib.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adducts import ELEM, PROTON, adduct_shift, expected_mz, formula_mass, within_tol

CANDIDATES = [a for a in [
    '[M+H]+', '[M+Na]+', '[M+K]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+',
    '[M]+', '[M+2H]2+', '[2M+H]+', '[2M+Na]+',
    '[M-H]-', '[M+Cl]-', '[M+CH2O2-H]-', '[M+C2H4O2-H]-', '[2M-H]-',
    '[M-CH3]-', '[M-2H]-',
] if adduct_shift(a) is not None]
POS = [a for a in CANDIDATES if a.endswith('+') or a.endswith('2+')]
NEG = [a for a in CANDIDATES if a.endswith('-')]
POL = {a: ('+' if a in POS else '-') for a in CANDIDATES}


def main():
    tr = pd.read_parquet('data/train.parquet', columns=[
        'ingest_lib', 'molecular_formula', 'ionization_mode', 'adduct',
        'adduct_orig', 'precursor_mz', 'precursor_error_ppm'])
    mmap = {f: formula_mass(f) for f in tr['molecular_formula'].dropna().unique()}
    tr['M'] = tr['molecular_formula'].map(mmap)
    tr['ion_formula'] = tr['molecular_formula'].str.endswith(('+', '-'))

    # expected mz under the LABELLED adduct (neutral-formula rows only)
    def exp_row(r):
        m = r.M
        if m is None or (isinstance(m, float) and np.isnan(m)):
            return np.nan
        return expected_mz(m, r.adduct)
    tr['exp_mz'] = [exp_row(r) for r in tr.itertuples()]
    tr['resid_da'] = tr['precursor_mz'] - tr['exp_mz']
    tr['tol_da'] = np.maximum(tr['exp_mz'] * 10e-6, 0.01)
    tr['label_ok'] = tr['resid_da'].abs() <= tr['tol_da']

    audit = tr[~tr['ion_formula'] & tr['exp_mz'].notna()].copy()
    print(f'rows auditable: {len(audit)}/{len(tr)} '
          f'({len(audit)/len(tr)*100:.2f}%)')
    print(f'label OK: {audit["label_ok"].mean()*100:.2f}%  '
          f'MISMATCH: {(~audit["label_ok"]).mean()*100:.2f}%')

    print('\n--- mismatch rate per ingest_lib ---')
    g = audit.groupby('ingest_lib')['label_ok'].agg(['mean', 'size'])
    g['mismatch_%'] = (1 - g['mean']) * 100
    print(g.sort_values('mismatch_%', ascending=False).to_string())

    print('\n--- mismatch rate per labelled adduct (top 15 by count) ---')
    g2 = audit.groupby('adduct')['label_ok'].agg(['mean', 'size'])
    g2['mismatch_%'] = (1 - g2['mean']) * 100
    print(g2.sort_values('size', ascending=False).head(15).to_string())

    # best-fit adduct search for mismatches (same polarity only)
    mis = audit[~audit['label_ok']].copy()
    cands_by_pol = {'+': POS, '-': NEG}
    best, best_res = [], []
    M = mis['M'].to_numpy()
    pmz = mis['precursor_mz'].to_numpy()
    pol = (mis['ionization_mode'] == 'negative').map({True: '-', False: '+'})
    for i in range(len(mis)):
        cands = cands_by_pol.get(pol.iloc[i], CANDIDATES)
        br, ba = None, None
        for a in cands:
            e = expected_mz(M[i], a)
            if e is None:
                continue
            r = abs(pmz[i] - e)
            if br is None or r < br:
                br, ba = r, a
        best.append(ba)
        best_res.append(br if br is not None else np.nan)
    mis['best_adduct'] = best
    mis['best_resid'] = best_res
    mis['best_tol'] = np.maximum(mis['precursor_mz'] * 10e-6, 0.01)
    mis['best_ok'] = mis['best_resid'] <= mis['best_tol']
    print(f'\nmismatches rescued by alternate adduct: {mis["best_ok"].mean()*100:.2f}% '
          f'({mis["best_ok"].sum()}/{len(mis)})')

    print('\n--- top label -> best-fit corrections ---')
    conf = mis[mis['best_ok']].groupby(['adduct', 'best_adduct']).size().reset_index(name='n')
    conf = conf.sort_values('n', ascending=False)
    print(conf.head(20).to_string(index=False))

    print('\n--- rescue rate per ingest_lib ---')
    print(mis.groupby('ingest_lib')['best_ok'].agg(['mean', 'sum', 'size'])
          .sort_values('mean', ascending=False).to_string())

    os.makedirs('data-clean', exist_ok=True)
    out = tr[['ingest_lib', 'molecular_formula', 'adduct', 'precursor_mz',
              'resid_da', 'label_ok']].copy()
    out['best_adduct'] = pd.Series([np.nan] * len(out), dtype='object')
    out['best_ok'] = pd.Series([np.nan] * len(out), dtype='object')
    out.loc[mis.index, 'best_adduct'] = list(mis['best_adduct'].values)
    out.loc[mis.index, 'best_ok'] = list(mis['best_ok'].values)
    out.to_parquet('data-clean/adduct_audit.parquet', index=False)
    print('\nwrote data-clean/adduct_audit.parquet', out.shape)


if __name__ == '__main__':
    main()
