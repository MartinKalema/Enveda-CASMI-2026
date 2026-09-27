"""Upgraded fragmentation: BDE-ordered cleavage, adduct-switched ionisation,
neutral-loss ladder. Pure RDKit+numpy. No learned weights.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# rough bond dissociation energies (kJ/mol) by bond class for ORDERING only
_BDE = {
    ("C", "C"): 347, ("C", "H"): 413, ("C", "N"): 305, ("C", "O"): 358,
    ("C", "S"): 259, ("C", "F"): 485, ("C", "Cl"): 327, ("C", "Br"): 285,
    ("C", "P"): 264, ("N", "H"): 391, ("O", "H"): 463, ("N", "N"): 163,
    ("O", "O"): 146, ("C", "C_arom"): 475,
}
# common neutral losses (name, mass)
NL_MENU = [
    ("H2O", 18.010565), ("CO", 27.994915), ("NH3", 17.026549),
    ("HCOOH", 46.005480), ("CH3OH", 32.026215), ("CO2", 43.989829),
    ("C2H4O2", 60.021130), ("hexose", 162.052824), ("pentose", 132.042259),
    ("deoxyhexose", 146.057909), ("HCl", 35.976678), ("HBr", 79.926160),
]
ADDUCT_DELTA = {
    "[M+H]+": 1.007276, "[M+Na]+": 22.989218, "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823, "[M-H]-": -1.007276, "[M+Cl]-": 34.968853,
    "[M+CH2O2-H]-": 44.998201, "[M+C2H4O2-H]-": 59.013851, "[M]+": 0.0,
}
# charge-site atoms per adduct regime: proton seeks N/O, sodium chelates O
CHARGE_ATOMS = {
    "pos": {"N", "O"}, "neg": {"O"}, "na": {"O"}, "nh4": {"N", "O"},
}


def _regime(adduct):
    a = adduct or ""
    if a in ("[M+Na]+", "[M+K]+"):
        return "na"
    if a in ("[M+NH4]+",):
        return "nh4"
    if a.endswith("]-"):
        return "neg"
    return "pos"


def _bond_bde(b, mol):
    a1 = mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol()
    a2 = mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()
    if b.GetBondType().name == "AROMATIC":
        return 475
    key = tuple(sorted((a1, a2)))
    return _BDE.get(key, _BDE.get((key[1], key[0]), 350))


def fragment_masses_bde(smiles, adduct="[M+H]+", max_breaks=2, top_bonds=14):
    """BDE-ordered cleavage seeded at charge-site atoms. Returns fragment masses."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    try:
        nm = m
        bonds = [(b.GetIdx(), _bond_bde(b, nm)) for b in nm.GetBonds()
                 if b.GetBondType() == Chem.BondType.SINGLE]
    except Exception:
        return []
    if not bonds:
        return []
    # charge-proximal first: score bonds by BDE + distance to charge atoms
    regime = _regime(adduct)
    wanted = CHARGE_ATOMS.get(regime, {"N", "O"})
    try:
        catoms = [a.GetIdx() for a in nm.GetAtoms() if a.GetSymbol() in wanted]
    except Exception:
        catoms = []
    import itertools
    scored = []
    for idx, bde in bonds:
        b = nm.GetBondWithIdx(idx)
        d = 0
        if catoms:
            d = min(abs(b.GetBeginAtomIdx() - c) + abs(b.GetEndAtomIdx() - c) for c in catoms)
        scored.append((bde + 8 * d, idx))
    scored.sort()
    keep = [i for _, i in scored[:top_bonds]]
    out = set()
    for k in (1, 2):
        if k > max_breaks or len(keep) < k:
            continue
        for combo in itertools.combinations(keep, k):
            try:
                frag = Chem.FragmentOnBonds(nm, list(combo), addDummies=False)
                for p in Chem.GetMolFrags(frag, asMols=True):
                    try:
                        Chem.SanitizeMol(p)
                        w = Descriptors.ExactMolWt(p)
                        out.add(round(w, 4))
                        out.add(round(w + 1.007825, 4))
                    except Exception:
                        continue
            except Exception:
                continue
    return sorted(out)


_CACHE = {}


def cached_fragments_bde(smiles, adduct="[M+H]+"):
    key = (smiles, adduct)
    hit = _CACHE.get(key)
    if hit is None:
        hit = np.array(fragment_masses_bde(smiles, adduct), dtype=float)
        _CACHE[key] = hit
    return hit


def frag_score_bde(mzs, intens, smiles, adduct="[M+H]+", prec_mz=None, tol=0.01, top_n=100):
    """sqrt-intensity fraction explained by BDE fragments AND neutral-loss menu.
    Returns (frag_frac, nl_frac)."""
    frags = cached_fragments_bde(smiles, adduct)
    d = ADDUCT_DELTA.get(adduct, 1.007276)
    mz = np.asarray(mzs, dtype=float)
    it = np.asarray(intens, dtype=float)
    o = np.argsort(-it)[:top_n]
    mz, it = mz[o], it[o]
    o2 = np.argsort(mz)
    mz, it = mz[o2], it[o2]
    w = np.sqrt(np.maximum(it, 0))
    tot = w.sum()
    if tot <= 0:
        return 0.0, 0.0
    hit_f = 0.0
    if len(frags):
        for m, wi in zip(mz, w):
            t = m - d
            j = int(np.searchsorted(frags, t))
            for jj in (j - 1, j):
                if 0 <= jj < len(frags) and abs(frags[jj] - t) <= tol:
                    hit_f += wi
                    break
    # neutral-loss ladder: precursor minus menu vs peaks
    hit_n = 0.0
    if prec_mz:
        for _, loss in NL_MENU:
            t = prec_mz - loss  # ion mass of [M-loss+adduct]
            j = int(np.searchsorted(mz, t)) if len(mz) else 0
            for jj in list(range(max(0, j - 2), min(len(mz), j + 3))):
                if abs(mz[jj] - t) / max(t, 1e-9) * 1e6 <= 15:
                    hit_n += w[jj]
                    break
    return float(hit_f / tot), float(min(hit_n / tot, 1.0))
