"""v4 channels: spectral-entropy similarity + analog propagation (RDKit-free runtime).
Entropy similarity (Li et al. 2021, matchms): low-entropy (few dominant peaks)
spectra match more selectively than cosine.
Analog propagation: score spectrum-less candidates (COCONUT) via spectrally
similar analogs WITH spectra: max_a sim(query,a)^p * Tanimoto(cand, analog).
"""
import numpy as np


def clean(mz, it, n_max=500, noise_pct=0.01):
    mz = np.asarray(mz, dtype=float); it = np.asarray(it, dtype=float)
    if len(mz) == 0: return mz, it
    keep = it >= it.max() * noise_pct
    mz, it = mz[keep], it[keep]
    if len(mz) > n_max:
        o = np.argsort(-it)[:n_max]
        mz, it = mz[o], it[o]
    o = np.argsort(mz)
    return mz[o], it[o]


def _match(mz1, it1, mz2, it2, tol=0.02):
    """Greedy matched peak pairs. Returns (a_matched, b_matched) intensity arrays."""
    i = j = 0
    A, B = [], []
    while i < len(mz1) and j < len(mz2):
        d = mz1[i] - mz2[j]
        if abs(d) <= tol:
            A.append(it1[i]); B.append(it2[j]); i += 1; j += 1
        elif d < 0: i += 1
        else: j += 1
    return np.array(A), np.array(B)


def _entropy(it):
    s = it.sum()
    if s <= 0 or len(it) == 0: return 0.0
    p = it / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def entropy_similarity(mz1, it1, mz2, it2, tol=0.02):
    """1 - (2*H_merged - H1 - H2)/ln(4) over matched peaks. 0..~1."""
    m1, i1 = clean(mz1, it1)
    m2, i2 = clean(mz2, it2)
    A, B = _match(m1, i1, m2, i2, tol)
    if len(A) < 3:
        return 0.0
    M = A + B
    s = 1.0 - (2 * _entropy(M) - _entropy(A) - _entropy(B)) / np.log(4)
    return float(max(0.0, min(1.0, s)))


def tanimoto(a, b):
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0
