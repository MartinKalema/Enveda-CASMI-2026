"""Train dedup audit: exact-duplicate spectra + same-structure multi-lab consistency.

Usage: .venv/bin/python v-clean/audit_dedup.py
(1) exact duplicates via spectrum hash (rounded m/z + intensity);
(2) per-structure spectrum counts and cross-library spread;
(3) same-structure same-adduct cosine consistency sample.
"""
import hashlib

import numpy as np
import pandas as pd


def spec_hash(mzs, ints, mz_dec=4, int_dec=3):
    m = np.round(np.asarray(mzs, dtype=float), mz_dec)
    v = np.round(np.asarray(ints, dtype=float), int_dec)
    o = np.argsort(m, kind='stable')
    h = hashlib.md5()
    h.update(m[o].tobytes())
    h.update(v[o].tobytes())
    return h.hexdigest()


def cosine(mz1, it1, mz2, it2, tol=0.02):
    a = np.asarray(it1, dtype=float)
    b = np.asarray(it2, dtype=float)
    na, nb = float(np.sqrt((a * a).sum())), float(np.sqrt((b * b).sum()))
    if na == 0 or nb == 0:
        return 0.0
    m1 = np.asarray(mz1, dtype=float)
    m2 = np.asarray(mz2, dtype=float)
    o1, o2 = np.argsort(m1), np.argsort(m2)
    m1, a, m2, b = m1[o1], a[o1], m2[o2], b[o2]
    i = j = 0
    num = 0.0
    while i < len(m1) and j < len(m2):
        d = m1[i] - m2[j]
        if abs(d) <= tol:
            num += a[i] * b[j]
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return num / (na * nb)


def main():
    tr = pd.read_parquet('data/train.parquet', columns=[
        'ingest_lib', 'normalized_smiles', 'inchikey14', 'adduct',
        'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
    print(f'rows: {len(tr)}  uniq smiles: {tr["normalized_smiles"].nunique()}  '
          f'uniq inchikey14: {tr["inchikey14"].nunique()}')

    tr['shash'] = [spec_hash(m, v) for m, v in
                   zip(tr['ms2_mzs'], tr['ms2_normalized_intensities'])]
    hc = tr['shash'].value_counts()
    dup_rows = (tr['shash'].map(hc) > 1).sum()
    print(f'\nexact-duplicate spectra: {dup_rows} rows '
          f'({dup_rows/len(tr)*100:.2f}%) in {int((hc > 1).sum())} hash groups')
    print('dup-group size distribution:',
          hc[hc > 1].describe([.5, .9, .99]).round(1).to_string())

    # are duplicates same-structure or collisions across structures?
    tr['hgrpcount'] = tr['shash'].map(hc)
    dups = tr[tr['hgrpcount'] > 1]
    per_hash_structs = dups.groupby('shash')['inchikey14'].nunique()
    print(f'\ndup groups spanning >1 structure: {(per_hash_structs > 1).sum()} '
          f'/ {len(per_hash_structs)} (hash collisions or true identical spectra)')
    same_struct = (per_hash_structs == 1).sum()
    print(f'dup groups within one structure: {same_struct}')
    per_hash_libs = dups.groupby('shash')['ingest_lib'].nunique()
    print(f'dup groups spanning >1 library: {(per_hash_libs > 1).sum()}')

    # spectra-per-structure distribution
    print('\n--- spectra per inchikey14 ---')
    spc = tr['inchikey14'].value_counts()
    print(spc.describe([.5, .9, .99]).round(1).to_string())
    print(f'singletons: {(spc == 1).sum()} ({(spc == 1).mean()*100:.1f}% of structures)')
    libs_per_struct = tr.groupby('inchikey14')['ingest_lib'].nunique()
    print(f'structures in >1 library: {(libs_per_struct > 1).sum()} '
          f'({(libs_per_struct > 1).mean()*100:.1f}%)')

    # same-structure consistency: sample 300 structures with >=2 spectra,
    # mean pairwise cosine within same adduct vs across adducts
    rng = np.random.default_rng(7)
    multi = tr.groupby('inchikey14').filter(lambda d: len(d) >= 2)
    structs = rng.choice(multi['inchikey14'].unique(), size=min(300, multi['inchikey14'].nunique()),
                         replace=False)
    same_ad, x_ad = [], []
    for s in structs:
        d = tr[tr['inchikey14'] == s].reset_index(drop=True)
        if len(d) > 8:
            d = d.sample(8, random_state=0).reset_index(drop=True)
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                c = cosine(d.loc[i, 'ms2_mzs'], d.loc[i, 'ms2_normalized_intensities'],
                           d.loc[j, 'ms2_mzs'], d.loc[j, 'ms2_normalized_intensities'])
                (same_ad if d.loc[i, 'adduct'] == d.loc[j, 'adduct'] else x_ad).append(c)
    print(f'\nsame-structure same-adduct pairs: {len(same_ad)} '
          f'mean cos {np.mean(same_ad):.3f} median {np.median(same_ad):.3f}')
    print(f'same-structure cross-adduct pairs: {len(x_ad)} '
          f'mean cos {np.mean(x_ad):.3f} median {np.median(x_ad):.3f}')

    tr[['shash']].to_parquet('data-clean/spec_hash.parquet', index=False)
    print('\nwrote data-clean/spec_hash.parquet')


if __name__ == '__main__':
    main()
