import math
import numpy as np, pandas as pd, torch, glob, os
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.Descriptors import ExactMolWt
HAVE_RDKIT = True
_g = {}
_MODEL = None
ADDUCT_LIST = ["[M+H]+","[M+NH4]+","[M+Na]+","[M+K]+","[M-H2O+H]+","[M-2H2O+H]+","[M]+",
               "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+C2H4O2-H]-","[M+Cl]-","[M]-",
               "[M+2H]2+","[M-2H]-","[2M+H]+","[2M+Na]+","[2M+NH4]+","[2M-H]-","[2M+K]+",
               "[2M+CH2O2-H]-","[2M+C2H4O2-H]-","[2M+Na-2H]-","[M+Na-2H]-","[M-H2O]+","<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
INSTR_IX = {a: i for i, a in enumerate(INSTR_LIST)}
def instr_family(s):
    if s is None: return 4
    t = str(s).lower()
    if 'timstof' in t: return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t: return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qq' in t: return 3
    return 4
MAX_PEAKS_NN = 128
class SinEmb(nn.Module):
    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
        self.register_buffer('inv', (2 * math.pi) / wav)
    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__(); self.h = h
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)
    def forward(self, x, pad):
        B, N, D = x.shape; y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))

class FPNet(nn.Module):
    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):
        super().__init__()
        self.d = d
        self.mz_emb = SinEmb(d)
        self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.gl = nn.Linear(d + 3, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat([self.prec_emb(prec),
                               (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks: x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([cls, mean], -1))

_MODEL = None

def load_neural_models():
    global _MODEL
    paths = sorted(glob.glob('/kaggle/input/**/fp_*.pt', recursive=True))
    if not paths:
        print("[INFO] No pretrained FPNet checkpoints found — proceeding with 3 channels.")
        return None
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    single, merged = [], []
    for pth in paths:
        ck = torch.load(pth, map_location='cpu', weights_only=False)
        net = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers']).to(dev).eval()
        net.load_state_dict(ck['model'])
        if 'merged' in pth.split('/')[-1]:
            merged.append(net)
        else:
            single.append(net)
        print(f"  Loaded {os.path.basename(pth)}: d={ck['d']}, layers={ck['layers']}, step={ck.get('step')}")
    print(f"[SUCCESS] Neural FPNet Ensemble ready: {len(single)} single + {len(merged)} merged on {dev}")
    _MODEL = (single, merged, dev, ck['nbits'])
    return _MODEL

def prep_peaks(mz, inten, prec_mz, max_peaks=MAX_PEAKS_NN, floor=1e-3, win=50.0, per_win=8):
    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = (mz <= prec_mz + 1.5)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}; sel = []
        for i in order:
            b = bucket[i]; c = cnt.get(b, 0)
            if c < per_win: cnt[b] = c + 1; sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            rest = np.array([i for i in order if i not in set(sel.tolist())])
            need = max_peaks - len(sel)
            if len(rest): sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)

def _merge_peaks(sub):
    mz = np.concatenate([np.asarray(r.ms2_mzs, float) for r in sub.itertuples()])
    it = np.concatenate([np.asarray(r.ms2_normalized_intensities, float) /
                         max(float(np.asarray(r.ms2_normalized_intensities, float).max()), 1e-9)
                         for r in sub.itertuples()])
    o = np.argsort(mz); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < 0.005:
            if it[j] >= it[j - 1]: keep[j - 1] = False
            else: keep[j] = False
    return mz[keep], it[keep]

@torch.no_grad()
def model_logits(sub):
    if _MODEL is None: return None
    single, merged, dev, nbits = _MODEL
    out = []
    if single:
        za = _logits_from(sub, single)
        if za is not None: out.append(za)
    if merged:
        mz, it = _merge_peaks(sub)
        r0 = next(sub.itertuples())
        zb = _logits_raw([(mz, it)], merged, float(np.median(sub.precursor_mz)), r0.adduct,
                         r0.instrument_type, 25.0,
                         float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode])))
        if zb is not None: out.append(zb)
    return np.mean(out, axis=0) if out else None

@torch.no_grad()
def _logits_from(sub, nets):
    if _MODEL is None: return None
    dev = _MODEL[2]
    rows = list(sub.itertuples())
    P = [prep_peaks(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz)) for r in rows]
    P = [(a, b) for a, b in P if len(a)]
    if not P: return None
    B = len(P); N = max(len(a) for a, _ in P)
    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
    for i, (a, b) in enumerate(P):
        mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False
    def ce_of(r):
        v = r.collision_energy_ev
        try: return float(np.mean(np.atleast_1d(v))) if v is not None and len(np.atleast_1d(v)) else 25.0
        except Exception: return 25.0
    T = lambda x: torch.as_tensor(x, device=dev)
    args = (T(mz), T(it), T(pad),
            T(np.array([float(r.precursor_mz) for r in rows[:B]], np.float32)),
            T(np.array([ADDUCT_IX.get(r.adduct, ADDUCT_IX['<unk>']) for r in rows[:B]])),
            T(np.array([instr_family(r.instrument_type) for r in rows[:B]])),
            T(np.array([ce_of(r) for r in rows[:B]], np.float32)),
            T(np.array([1.0 if r.ionization_mode == 'positive' else -1.0 for r in rows[:B]], np.float32)))
    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)

@torch.no_grad()
def _logits_raw(pairs, nets, prec, adduct, instrument, ce, mode):
    if _MODEL is None: return None
    dev = _MODEL[2]
    P = [prep_peaks(mz, it, prec) for mz, it in pairs]
    P = [(a, b) for a, b in P if len(a)]
    if not P: return None
    B = len(P); N = max(len(a) for a, _ in P)
    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
    for i, (a, b) in enumerate(P):
        mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False
    T = lambda x: torch.as_tensor(x, device=dev)
    args = (T(mz), T(it), T(pad), T(np.full(B, prec, np.float32)),
            T(np.full(B, ADDUCT_IX.get(adduct, ADDUCT_IX['<unk>']))),
            T(np.full(B, instr_family(instrument))),
            T(np.full(B, ce, np.float32)), T(np.full(B, mode, np.float32)))
    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)

def _fp_init():
    if not HAVE_RDKIT: return
    _g['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def fp_and_mass(smi):
    if not HAVE_RDKIT: return None
    if not _g: _fp_init()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([
            _g['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
            _g['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
            _g['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
            np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)
        ])[BITS]
        return fp, float(ExactMolWt(m))
    except Exception:
        return None
