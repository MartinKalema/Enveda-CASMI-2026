"""Formula-mass index rebuild + 20-query retrieval probe.

Usage: .venv/bin/python v-clean/probe.py
(1) builds exact-mass-keyed formula index -> data-clean/formula_mass_index.parquet;
    quantifies 10ppm mass-collision ambiguity;
(2) 20-query probe: top-1/top-5 correct-structure hit rate by cosine,
    raw vs denoised (0.01 + top200), and mass-window recall with
    labelled vs corrected adduct.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adducts import expected_mz, formula_mass
from audit_peaks import cosine, denoise


def build_index():
    tr = pd.read_parquet('data/train.parquet', columns=['molecular_formula'])
    formulae = tr['molecular_formula'].dropna().unique()
    rows = [(f, formula_mass(f)) for f in formulae]
    idx = pd.DataFrame(rows, columns=['formula', 'exact_mass'])
    idx['parse_ok'] = idx['exact_mass'].notna()
    print(f'unique formulae: {len(idx)}  parse-ok: {idx["parse_ok"].sum()} '
          f'({idx["parse_ok"].mean()*100:.2f}%)')
    ok = idx[idx['parse_ok']].sort_values('exact_mass').reset_index(drop=True)
    # 10ppm collision: neighbours within tolerance
    tol = ok['exact_mass'] * 10e-6
    d = np.diff(ok['exact_mass'].to_numpy())
    t = tol.to_numpy()
    allow = (t[:-1] + t[1:]) / 2
    n_coll = int((d <= allow).sum())
    print(f'formulae with a 10ppm neighbour: {n_coll} ({n_coll/len(ok)*100:.2f}%)')
    # nominal-mass collisions
    ok['nominal'] = ok['exact_mass'].round().astype(int)
    multi = ok.groupby('nominal')['formula'].nunique()
    print(f'nominal masses shared by >1 formula: {(multi > 1).sum()}/{len(multi)} '
          f'max formulae per nominal mass: {multi.max()}')
    os.makedirs('data-clean', exist_ok=True)
    ok.to_parquet('data-clean/formula_mass_index.parquet', index=False)
    print('wrote data-clean/formula_mass_index.parquet', ok.shape)
    return ok


def run_probe():
    tr = pd.read_parquet('data/train.parquet', columns=[
        'normalized_smiles', 'inchikey14', 'molecular_formula', 'adduct', 'precursor_mz',
        'ms2_mzs', 'ms2_normalized_intensities'])
    au = pd.read_parquet('data-clean/adduct_audit.parquet',
                         columns=['label_ok', 'best_adduct', 'best_ok'])
    tr = pd.concat([tr.reset_index(drop=True), au.reset_index(drop=True)], axis=1)
    rng = np.random.default_rng(11)
    # 20 query structures with >=3 spectra, [M+H]+ available
    cand_structs = tr.groupby('inchikey14').filter(
        lambda d: len(d) >= 3 and (d['adduct'] == '[M+H]+').any())['inchikey14'].unique()
    qstructs = rng.choice(cand_structs, size=20, replace=False)
    # pool: 2000 spectra from other structures + the 20 truths excluded from pool
    # (structure-disjoint scoring: truth = same inchikey14)
    pool = tr[~tr['inchikey14'].isin(qstructs)].sample(2000, random_state=2).reset_index(drop=True)
    pool_dn = [denoise(m, v) for m, v in
               zip(pool['ms2_mzs'], pool['ms2_normalized_intensities'])]

    res = []
    for s in qstructs:
        q = tr[(tr['inchikey14'] == s) & (tr['adduct'] == '[M+H]+')].iloc[0]
        q_dn = denoise(q['ms2_mzs'], q['ms2_normalized_intensities'])
        scores_raw, scores_dn = [], []
        for i, r in pool.iterrows():
            scores_raw.append(cosine(q['ms2_mzs'], q['ms2_normalized_intensities'],
                                     r['ms2_mzs'], r['ms2_normalized_intensities']))
            scores_dn.append(cosine(q_dn[0], q_dn[1], pool_dn[i][0], pool_dn[i][1]))
        # inject one truth spectrum (different spectrum of same structure) into ranking
        truth = tr[(tr['inchikey14'] == s)].iloc[1]
        t_raw = cosine(q['ms2_mzs'], q['ms2_normalized_intensities'],
                       truth['ms2_mzs'], truth['ms2_normalized_intensities'])
        t_dn_mz, t_dn_i = denoise(truth['ms2_mzs'], truth['ms2_normalized_intensities'])
        t_dn = cosine(q_dn[0], q_dn[1], t_dn_mz, t_dn_i)
        rank_raw = 1 + int(np.sum(np.asarray(scores_raw) >= t_raw))
        rank_dn = 1 + int(np.sum(np.asarray(scores_dn) >= t_dn))
        # mass-window check: labelled vs corrected adduct neutral mass vs truth formula mass
        fm = formula_mass(q['molecular_formula'])
        e_lab = expected_mz(fm, q['adduct']) if fm else None
        lab_ok = (e_lab is not None and abs(q['precursor_mz'] - e_lab)
                  <= max(e_lab * 10e-6, 0.01))
        bad = q['best_adduct']
        e_fix = expected_mz(fm, bad) if (fm and isinstance(bad, str)) else None
        fix_ok = (e_fix is not None and abs(q['precursor_mz'] - e_fix)
                  <= max(e_fix * 10e-6, 0.01))
        res.append({'struct': s, 'rank_raw': rank_raw, 'rank_dn': rank_dn,
                    't_raw': round(t_raw, 3), 't_dn': round(t_dn, 3),
                    'masswin_lab': lab_ok, 'masswin_fixed': fix_ok,
                    'q_best': bad if not lab_ok else None})
    r = pd.DataFrame(res)
    print('\n--- 20-query probe (pool=2000, structure-disjoint) ---')
    print(f'top-1: raw {(r["rank_raw"] == 1).sum()}/20  '
          f'denoised {(r["rank_dn"] == 1).sum()}/20')
    print(f'top-5: raw {(r["rank_raw"] <= 5).sum()}/20  '
          f'denoised {(r["rank_dn"] <= 5).sum()}/20')
    print(f'median rank: raw {r["rank_raw"].median():.0f}  '
          f'denoised {r["rank_dn"].median():.0f}')
    print(f'mean truth-cosine: raw {r["t_raw"].mean():.3f}  '
          f'denoised {r["t_dn"].mean():.3f}')
    print(f'queries with bad adduct label: {(~r["masswin_lab"]).sum()}/20, '
          f'rescued by correction: '
          f'{((~r["masswin_lab"]) & r["masswin_fixed"]).sum()}/20')
    print(r.to_string(index=False))
    r.to_csv('data-clean/probe20.csv', index=False)
    print('\nwrote data-clean/probe20.csv')


def main():
    build_index()
    run_probe()


if __name__ == '__main__':
    main()
