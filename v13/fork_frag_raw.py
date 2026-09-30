import numpy as np
from multiprocessing import Pool as MPool
from numba import njit, prange
HAVE_RDKIT = True
INT_FLOOR = 0.002
MAX_PEAKS = 256
INT_POWER = 1.0
MZ_TOL = 0.01
from rdkit import Chem
from rdkit.Chem import Descriptors
from types import SimpleNamespace
CFG = SimpleNamespace(INT_FLOOR=0.002, MAX_PEAKS=256, INT_POWER=1.0,
                      ENT_WEIGHT=True, MZ_TOL=0.01)

AMU = {'C':12.0, 'H':1.00782503207, 'N':14.0030740048, 'O':15.9949146196, 'P':30.97376163,
       'S':31.97207100, 'F':18.99840322, 'Cl':34.96885268, 'Br':78.9183371, 'I':126.904473,
       'Na':22.9897692809, 'K':38.96370668, 'Si':27.9769265325, 'B':11.0093054, 'Se':79.9165213}
H_ATOM = AMU['H']
PROTON_MASS = H_ATOM - 0.00054857990

def clean_spectrum(mz, it):
    return _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                  CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)

def _clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx: mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr: c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c): v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk): k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float32)
    oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        v = it[idx[i]] ** power
        oi[i] = v
        s += v
    if s > 0:
        for i in range(c): oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0: S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                oi[i] = oi[i] ** w
                s2 += oi[i]
            if s2 > 0:
                for i in range(c): oi[i] /= s2
    return om, oi

def mol_graph(smi):
    if not HAVE_RDKIT: return None
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    n = m.GetNumAtoms()
    w = np.zeros(n)
    for a in m.GetAtoms():
        w[a.GetIdx()] = AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H_ATOM
    if (w == 0).any(): return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n

def _components(n, bonds, drop):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop: continue
        adj[a].append(b); adj[b].append(a)
    seen = np.zeros(n, bool); comps = []
    for s in range(n):
        if seen[s]: continue
        stack = [s]; seen[s] = True; cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]: seen[v] = True; stack.append(v); cur.append(v)
        comps.append(cur)
    return comps

def fragment_masses(smi, max_breaks=2, max_bonds=34):
    g = mol_graph(smi)
    if g is None: return np.zeros(0)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds: return np.array([w.sum()])
    out = {w.sum()}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    return np.array(sorted(out))

def explain_score(frag_mass, peak_mz, peak_int, mode=1.0, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):
    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0
    ion = []
    for dh in h_shifts:
        ion.append(frag_mass + dh * H_ATOM + (PROTON_MASS if mode > 0 else -PROTON_MASS))
    ion = np.sort(np.concatenate(ion))
    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()
    if tot <= 0: return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)

def _frag_masses_wrapper(smi):
    try: return fragment_masses(smi)
    except Exception: return np.zeros(0)

def frag_scores(cand_smiles, specs, mode, workers=4):
    if not HAVE_RDKIT:
        return np.zeros(len(cand_smiles), np.float32)
    with MPool(workers) as mp:
        frags = mp.map(_frag_masses_wrapper, cand_smiles, chunksize=8)
    peaks = []
    for mz, it, *_ in specs:
        m2, i2 = _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                        CFG.INT_FLOOR, CFG.MAX_PEAKS, 1.0, False)
        peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
    out = np.zeros(len(cand_smiles), np.float32)
    for j, f in enumerate(frags):
        out[j] = max((explain_score(f, a, b, mode=mode, tol=CFG.MZ_TOL) for a, b in peaks), default=0.0)
    return out

# ===================================================================================
#  Channel 4: Neural Spectrum -> Molecular Fingerprint Model (FPNet)
#  Ranking is f . z (Exact Bayes log-likelihood dot product)
# ===================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_PEAKS_NN = 128
ADDUCT_LIST = ["[M+H]+","[M+NH4]+","[M+Na]+","[M+K]+","[M-H2O+H]+","[M-2H2O+H]+","[M]+",
               "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+C2H4O2-H]-","[M+Cl]-","[M]-",
               "[M+2H]2+","[M-2H]-","[2M+H]+","[2M+Na]+","[2M+NH4]+","[2M-H]-","[2M+K]+",
               "[2M+CH2O2-H]-","[2M+C2H4O2-H]-","[2M+Na-2H]-","[M+Na-2H]-","[M-H2O]+","<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
INSTR_IX = {a: i for i, a in enumerate(INSTR_LIST)}
