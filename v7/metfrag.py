"""MetFrag-lite: 1-2 bond cleavage fragment explanation (RDKit).
Score = sqrt-intensity fraction of peaks explained by substructure fragments.
Independent evidence (survey: corr 0.058 with analog channel).
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

ADDUCT_DELTA = {
    "[M+H]+": 1.007276, "[M+Na]+": 22.989218, "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823, "[M-H]-": -1.007276, "[M+Cl]-": 34.968853,
    "[M+CH2O2-H]-": 44.998201, "[M+C2H4O2-H]-": 59.013851, "[M]+": 0.0,
}


def fragment_masses(smiles, max_bonds=2):
    """Neutral monoisotopic masses of 1-2 bond cleavage fragments (+H rearrangements)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None: return []
    m = Chem.AddHs(m)
    bonds = [b.GetIdx() for b in m.GetBonds()
             if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
    out = set()
    import itertools
    try:
        nm = Chem.RemoveHs(m)
    except Exception:
        return []
    for k in (1, 2):
        if k > max_bonds or len(bonds) < k: continue
        for combo in itertools.combinations(bonds, k):
            try:
                frag = Chem.FragmentOnBonds(nm, list(combo), addDummies=False)
                parts = Chem.GetMolFrags(frag, asMols=True)
                for p in parts:
                    try:
                        Chem.SanitizeMol(p)
                        out.add(round(Descriptors.ExactMolWt(p), 4))
                        out.add(round(Descriptors.ExactMolWt(p) + 1.007825, 4))  # +H
                    except Exception:
                        continue
            except Exception:
                continue
    return sorted(out)


_FRAG_CACHE = {}


def cached_fragments(smiles, max_bonds=2):
    key = (smiles, max_bonds)
    hit = _FRAG_CACHE.get(key)
    if hit is None:
        hit = np.array(sorted(fragment_masses(smiles, max_bonds)), dtype=float)
        _FRAG_CACHE[key] = hit
    return hit


def frag_score(mzs, intens, smiles, adduct="[M+H]+", tol=0.01, top_n=100):
    """sqrt-intensity fraction explained. Returns 0..~1."""
    frags = cached_fragments(smiles)
    frags = np.asarray(frags, dtype=float)
    if len(frags) == 0: return 0.0
    d = ADDUCT_DELTA.get(adduct, 1.007276)
    mzs = np.asarray(mzs, dtype=float); intens = np.asarray(intens, dtype=float)
    o = np.argsort(-intens)[:top_n]
    mzs, intens = mzs[o], intens[o]
    w = np.sqrt(np.maximum(intens, 0))
    tot = w.sum()
    if tot <= 0: return 0.0
    frags = np.array(sorted(frags))
    hit = 0.0
    for mz, wi in zip(mzs, w):
        target = mz - d
        j = int(np.searchsorted(frags, target))
        ok = False
        for jj in (j - 1, j):
            if 0 <= jj < len(frags) and abs(frags[jj] - target) <= tol:
                ok = True; break
        if ok: hit += wi
    return float(hit / tot)
