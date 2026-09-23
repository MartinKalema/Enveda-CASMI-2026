"""Adduct-shift table + formula exact mass. Shared by audits and runtime cleaning."""
import re

# monoisotopic masses
ELEM = {'H': 1.00782503223, 'C': 12.0, 'N': 14.00307400443, 'O': 15.99491461957,
        'P': 30.9737619985, 'S': 31.9720711744, 'F': 18.99840316273,
        'Cl': 34.968852682, 'Br': 78.9183376, 'I': 126.9044719,
        'Na': 22.989769282, 'K': 38.9637074864, 'Si': 27.97692653465,
        'B': 11.00930536, 'Se': 79.9165218, 'Zn': 63.9291422, 'Fe': 55.9349363,
        'Cu': 62.9295975, 'Mg': 23.985041697, 'Ca': 39.962590983, 'Li': 7.0160034366,
        'Al': 26.98153853, 'Mn': 54.93804391, 'Co': 58.93319505, 'Ni': 57.9353429,
        'As': 74.92159457, 'D': 2.01410177812}
PROTON = 1.00727646662
ELECTRON = 0.000548579909

FORMULA_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


def formula_mass(fstr):
    """Monoisotopic exact mass of a neutral formula string. None if unparseable."""
    if not isinstance(fstr, str) or not fstr:
        return None
    total = 0.0
    pos = 0
    for m in FORMULA_RE.finditer(fstr):
        if m.start() != pos:
            return None  # gap -> unparseable (brackets, charges, dots)
        el, n = m.group(1), int(m.group(2)) if m.group(2) else 1
        if el not in ELEM:
            return None
        total += ELEM[el] * n
        pos = m.end()
    if pos != len(fstr):
        return None
    return total if total > 0 else None


def adduct_shift(adduct):
    """Return (multiplier m, additive shift Da, charge z) s.t. mz = (m*M + shift)/z."""
    if not isinstance(adduct, str):
        return None
    a = adduct.strip()
    H = PROTON
    table = {
        '[M+H]+': (1, H, 1),
        '[M+Na]+': (1, ELEM['Na'] - ELECTRON, 1),
        '[M+K]+': (1, ELEM['K'] - ELECTRON, 1),
        '[M+NH4]+': (1, ELEM['N'] + 4 * ELEM['H'] - ELECTRON, 1),
        '[M-H]-': (1, -H, 1),
        '[M+Cl]-': (1, ELEM['Cl'] + ELECTRON, 1),
        '[M+CH2O2-H]-': (1, ELEM['C'] + 2 * ELEM['H'] + 2 * ELEM['O'] - H, 1),
        '[M+C2H4O2-H]-': (1, 2 * ELEM['C'] + 4 * ELEM['H'] + 2 * ELEM['O'] - H, 1),
        '[2M+H]+': (2, H, 1),
        '[2M+Na]+': (2, ELEM['Na'] - ELECTRON, 1),
        '[2M-H]-': (2, -H, 1),
        '[2M+Na-2H]-': (2, ELEM['Na'] - 2 * ELEM['H'] + ELECTRON, 1),
        '[2M+CH2O2-H]-': (2, ELEM['C'] + 2 * ELEM['H'] + 2 * ELEM['O'] - H, 1),
        '[M-H2O+H]+': (1, -2 * ELEM['H'] - ELEM['O'] + H, 1),
        '[M-2H2O+H]+': (1, -2 * (2 * ELEM['H'] + ELEM['O']) + H, 1),
        '[M-H2O]+': (1, -2 * ELEM['H'] - ELEM['O'] - ELECTRON, 1),
        '[M]+': (1, -ELECTRON, 1),
        '[M+2H]2+': (1, 2 * H, 2),
        '[M-CH3]-': (1, -(ELEM['C'] + 3 * ELEM['H']) + ELECTRON, 1),
        '[M-2H]-': (1, -2 * ELEM['H'] + ELECTRON, 1),
        '[M+CH3COO]-': (1, 2 * ELEM['C'] + 3 * ELEM['H'] + 2 * ELEM['O'] + ELECTRON, 1),
    }
    if a in table:
        return table[a]
    return parse_adduct_generic(a)


ADDUCT_RE = re.compile(r'^\[(\d*)M((?:[+-][A-Za-z0-9]+)*)\](\d*)([+-])$')
FRAG_RE = re.compile(r'([+-])((?:\d+)?[A-Za-z][A-Za-z0-9]*)')
LEADNUM_RE = re.compile(r'^(\d+)(.*)$')


def parse_adduct_generic(adduct):
    """Compositional parse of '[{m}M{frags}]{z}{sign}'. Returns (m, shift, z)."""
    if not isinstance(adduct, str):
        return None
    mt = ADDUCT_RE.match(adduct.strip())
    if not mt:
        return None
    mult = int(mt.group(1)) if mt.group(1) else 1
    frags, zstr, sign = mt.group(2), mt.group(3), mt.group(4)
    if not frags:
        # bare [M]+ / [M]- / [2M]2+ ...
        z = int(zstr) if zstr else 1
        shift = (-ELECTRON if sign == '+' else ELECTRON) * z
        return (mult, shift, z)
    total = 0.0
    for fm in FRAG_RE.finditer(frags):
        fstr = fm.group(2)
        lm = LEADNUM_RE.match(fstr)
        count, comp = (int(lm.group(1)), lm.group(2)) if lm else (1, fstr)
        fmass = formula_mass(comp)
        if fmass is None:
            return None
        total += count * fmass if fm.group(1) == '+' else -count * fmass
    z = int(zstr) if zstr else 1
    total += (-ELECTRON if sign == '+' else ELECTRON) * z
    return (mult, total, z)


def ion_mass(fstr):
    """Mass of a charged ion formula like 'C15H11O6+' (trailing charge stripped,
    electron mass neglected: good to ~0.5 mDa, inside our 10 mDa floor)."""
    if isinstance(fstr, str) and fstr and fstr[-1] in '+-':
        return formula_mass(fstr[:-1])
    return formula_mass(fstr)


def expected_mz(neutral_mass, adduct):
    p = adduct_shift(adduct)
    if p is None or neutral_mass is None:
        return None
    m, shift, z = p
    return (m * neutral_mass + shift) / z


def residual_da(precursor_mz, neutral_mass, adduct):
    e = expected_mz(neutral_mass, adduct)
    if e is None or precursor_mz is None:
        return None
    return precursor_mz - e


def within_tol(resid_da, expected_mz_val, ppm=10.0, abs_da=0.01):
    return abs(resid_da) <= max(expected_mz_val * ppm * 1e-6, abs_da)
