"""Extract the fork's frag channel (verified boundaries) for head-to-head tests."""
import json
import re

P = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
nb = json.load(open(f"{P}/kernels/fork-034/notebook.ipynb"))
full = ""
for c in nb["cells"]:
    if c["cell_type"] == "code":
        full += "".join(c["source"]) + "\n"


def block(s, start, stops):
    i = s.index(start)
    j = len(s)
    for st in stops:
        k = s.find(st, i + 1)
        if k != -1:
            j = min(j, k)
    return s[i:j]


out_head = ("import numpy as np\nfrom multiprocessing import Pool as MPool\n"
            "from numba import njit, prange\n"
            "HAVE_RDKIT = True\nINT_FLOOR = 0.002\nMAX_PEAKS = 256\n"
            "INT_POWER = 1.0\nMZ_TOL = 0.01\n"
            "from rdkit import Chem\nfrom rdkit.Chem import Descriptors\n"
            "from types import SimpleNamespace\n"
            "CFG = SimpleNamespace(INT_FLOOR=0.002, MAX_PEAKS=256, INT_POWER=1.0,\n"
            "                      ENT_WEIGHT=True, MZ_TOL=0.01)\n")
am = block(full, "AMU = {", ["\ndef clean_spectrum"])
cs = block(full, "def clean_spectrum", ["\ndef lib_sim"])
uc = block(full, "def _clean", ["\ndef entropy_sim"])
mg = block(full, "def mol_graph", ["\ndef _components"])
cp = block(full, "def _components", ["\ndef fragment_masses"])
fm = block(full, "def fragment_masses", ["\ndef explain_score"])
es = block(full, "def explain_score", ["\ndef _frag_masses_wrapper"])
fw = block(full, "def _frag_masses_wrapper", ["\ndef frag_scores"])
fs = block(full, "def frag_scores", ["\ndef instr_family"])
for name, b in [("am", am), ("cs", cs), ("uc", uc), ("mg", mg),
                ("cp", cp), ("fm", fm), ("es", es), ("fw", fw), ("fs", fs)]:
    assert "def main(" not in b and "__main__" not in b, name
    assert "find_file" not in b and "BITS" not in b, name
parts = [out_head, am, cs, uc, mg, cp, fm, es, fw, fs]
open(f"{P}/v13/fork_frag_raw.py", "w").write("\n".join(parts))
print("written", sum(map(len, parts)))
