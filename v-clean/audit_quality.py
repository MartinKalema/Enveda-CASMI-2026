"""Label-quality filter from precursor_error_ppm + adduct residuals.

Usage: .venv/bin/python v-clean/audit_quality.py
Quantifies what fraction of train spectra are mislabeled/noisy and proposes
hard filter thresholds for retrieval indexing.
"""
import os

import numpy as np
import pandas as pd

CUTS = [5, 10, 20, 50, 100, 1000]


def main():
    tr = pd.read_parquet('data/train.parquet', columns=[
        'ingest_lib', 'adduct', 'precursor_mz', 'precursor_error_ppm'])
    au = pd.read_parquet('data-clean/adduct_audit.parquet',
                         columns=['resid_da', 'label_ok', 'best_ok'])
    tr = pd.concat([tr.reset_index(drop=True), au.reset_index(drop=True)], axis=1)
    ae = tr['precursor_error_ppm'].abs()
    print(f'precursor_error_ppm nulls: {tr["precursor_error_ppm"].isna().sum()} '
          f'({tr["precursor_error_ppm"].isna().mean()*100:.3f}%)')
    print('\n--- |precursor_error_ppm| cumulative exclusion ---')
    for c in CUTS:
        print(f'  > {c:>5} ppm excluded: {(ae > c).mean()*100:7.3f}% '
              f'({(ae > c).sum()})')
    print('\n--- exclusion at 20ppm per ingest_lib ---')
    g = tr.groupby('ingest_lib')['precursor_error_ppm'].apply(
        lambda s: (s.abs() > 20).mean() * 100)
    n = tr.groupby('ingest_lib').size()
    print(pd.DataFrame({'n': n, 'pct_gt20ppm': g})
          .sort_values('pct_gt20ppm', ascending=False).to_string())

    # cross-tab: our adduct-mismatch flag vs precursor_error>20ppm
    tr['big_err'] = ae > 20
    print('\n--- adduct label_ok vs |precursor_error|>20ppm ---')
    print(pd.crosstab(tr['label_ok'], tr['big_err'], normalize='index')
          .round(4).to_string())
    print('\n--- P(label mismatch | big precursor err), per lib ---')
    sub = tr[~tr['ion_formula'] if 'ion_formula' in tr else slice(None)]
    print(tr.groupby('ingest_lib').apply(
        lambda d: pd.Series({
            'P(mis|err>20)': ((~d['label_ok']) & d['big_err']).sum() / max(d['big_err'].sum(), 1),
            'P(mis|err<=20)': ((~d['label_ok']) & ~d['big_err']).sum() / max((~d['big_err']).sum(), 1),
        }), include_groups=False).round(4).to_string())

    # proposed filter: exclude |err|>20ppm OR adduct-mismatch-unrescued
    tr['drop_20ppm'] = tr['big_err'].fillna(False)
    tr['drop_adduct'] = (~tr['label_ok'].fillna(True)) & (tr['best_ok'] != True)
    tr['drop_either'] = tr['drop_20ppm'] | tr['drop_adduct']
    print(f'\nproposed drops: >20ppm {tr["drop_20ppm"].mean()*100:.3f}%, '
          f'adduct-bad {tr["drop_adduct"].mean()*100:.3f}%, '
          f'either {tr["drop_either"].mean()*100:.3f}%')
    print(tr.groupby('ingest_lib')['drop_either'].mean().sort_values(
        ascending=False).round(4).to_string())

    os.makedirs('data-clean', exist_ok=True)
    tr[['precursor_error_ppm', 'drop_20ppm', 'drop_adduct',
        'drop_either']].to_parquet('data-clean/quality_flags.parquet', index=False)
    print('\nwrote data-clean/quality_flags.parquet')


if __name__ == '__main__':
    main()
