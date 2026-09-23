"""Peak-count / denoising audit.

Usage: .venv/bin/python v-clean/audit_peaks.py
(1) peak-count distributions train vs test, per ingest_lib;
(2) low-intensity peak fractions at candidate thresholds;
(3) self-cosine of spectra before/after denoising (information kept);
(4) effect of denoising on a 20-query retrieval probe is in probe.py.
"""
import os

import numpy as np
import pandas as pd


def frac_below(ints, thr):
    ints = np.asarray(ints, dtype=float)
    if len(ints) == 0:
        return 1.0
    return float((ints < thr).mean())


def denoise(mzs, ints, thr=0.01, topn=200):
    mzs = np.asarray(mzs, dtype=float)
    ints = np.asarray(ints, dtype=float)
    keep = ints >= thr
    mzs, ints = mzs[keep], ints[keep]
    if len(ints) > topn:
        idx = np.argsort(ints)[-topn:]
        mzs, ints = mzs[idx], ints[idx]
    o = np.argsort(mzs)
    return mzs[o], ints[o]


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
        'ingest_lib', 'ms2_mzs', 'ms2_normalized_intensities'])
    te = pd.read_parquet('data/test.parquet', columns=[
        'ms2_mzs', 'ms2_normalized_intensities'])
    tr['n'] = tr['ms2_mzs'].apply(len)
    te['n'] = te['ms2_mzs'].apply(len)
    print(f'train peaks: median {tr["n"].median():.0f} mean {tr["n"].mean():.0f} '
          f'p90 {tr["n"].quantile(.9):.0f} p99 {tr["n"].quantile(.99):.0f}')
    print(f'test  peaks: median {te["n"].median():.0f} mean {te["n"].mean():.0f} '
          f'p90 {te["n"].quantile(.9):.0f} p99 {te["n"].quantile(.99):.0f}')
    print('\n--- median/mean peaks per ingest_lib ---')
    print(tr.groupby('ingest_lib')['n'].agg(['median', 'mean', 'size'])
          .sort_values('median').round(1).to_string())

    rng = np.random.default_rng(0)
    for name, df in [('train', tr), ('test', te)]:
        samp = df.sample(min(len(df), 20000), random_state=0)
        for thr in [0.001, 0.005, 0.01, 0.02, 0.05]:
            f = samp['ms2_normalized_intensities'].apply(lambda v: frac_below(v, thr))
            print(f'{name}: frac peaks < {thr}: mean {f.mean()*100:.1f}%')
        # kept-count under thr=0.01 + top200 cap
        kept = samp.apply(lambda r: len(denoise(r['ms2_mzs'],
                                                r['ms2_normalized_intensities'])[0]), axis=1)
        print(f'{name}: kept peaks after 0.01+top200: median {kept.median():.0f} '
              f'mean {kept.mean():.1f}')

    # self-cosine: how much signal survives denoising
    print('\n--- self-cosine(raw, denoised) on 2000 train spectra ---')
    samp = tr.sample(2000, random_state=1)
    for thr, topn in [(0.01, 200), (0.02, 100), (0.005, 300)]:
        cs = [cosine(r.ms2_mzs, r.ms2_normalized_intensities,
                     *denoise(r.ms2_mzs, r.ms2_normalized_intensities, thr, topn))
              for r in samp.itertuples()]
        print(f'thr={thr} top{topn}: mean self-cos {np.mean(cs):.4f} '
              f'min {np.min(cs):.4f}')

    # intensity renormalization check: does test already differ in scale?
    print('\n--- max intensity == 1.0 fraction (renormalized?) ---')
    for name, df in [('train', tr), ('test', te)]:
        mx = df['ms2_normalized_intensities'].apply(
            lambda v: max(v) if len(v) else float('nan'))
        print(f'{name}: max==1.0 in {(np.abs(mx - 1.0) < 1e-9).mean()*100:.1f}% '
              f'of spectra; median max {mx.median():.3f}')


if __name__ == '__main__':
    main()
