"""Build kernels/baseline-v0/notebook.ipynb from v14 sources.
Matches submit_v14.py's actual imports. Usage: .venv/bin/python kernels/build_v14.py
"""
import json
import re

P = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def block(s, start, stops):
    i = s.index(start)
    mm = []
    for st in stops:
        for m in re.finditer(re.escape(st), s[i + 10:]):
            mm.append(m.start())
    if not mm:
        raise ValueError(f"no stops {stops} after {start[:40]}")
    return s[i:i + 10 + min(mm)]


sub = open(f"{P}/v1/subformula.py").read()
ble = open(f"{P}/v2/blend.py").read()
lbs_src = open(f"{P}/v14/libsearch.py").read()
lod_src = open(f"{P}/v14/loaders.py").read()
rnk_src = open(f"{P}/v14/rank.py").read()
fpf = open(f"{P}/v13/fork_fpnet_raw.py").read()
frg_src = open(f"{P}/v7/submit_v7.py").read()
frag_src = open(f"{P}/v13/fork_frag_raw.py").read()
run = open(f"{P}/v14/submit_v14.py").read()

ble = ble.replace("from v1.subformula import ADDUCT_DELTA\n", "")
ble = ble.replace(f'PROJECT = "{P}"', 'PROJECT = "."  # unused in kernel')
lbs = lbs_src + "\n" + lod_src.replace("import numpy as np\n", "")
lbs += "\nlib_neutral_mass = neutral_mass\n"
rnk = rnk_src
fpfrag = "\n".join([
    block(fpf, "class SinEmb", ["\nclass Block"]),
    block(fpf, "class Block", ["\nclass FPNet"]),
    block(fpf, "class FPNet", ["\ndef _merge_peaks"]),
    block(fpf, "def _merge_peaks", ["\ndef model_logits"]),
    block(fpf, "def model_logits", ["\ndef _logits_from"]),
    block(fpf, "def _logits_from", ["\ndef _logits_raw"]),
    block(fpf, "def _logits_raw", ["\ndef _fp_init"])])
_frag_stop = ["\ndef submit"] if "\ndef submit" in frg_src else ["\ndef main("]
frg = ("import pickle as _pk\nFRAG_TOL = 0.01\n"
       + block(frg_src, 'AD = {"[M+H]+', ["def neutral_mass"])
       + block(frg_src, "def frag_match", _frag_stop)
       + "\n" + block(frag_src, "AMU = {", ["\ndef clean_spectrum"])
       + "\n" + block(frag_src, "def explain_score", ["\ndef _frag_masses_wrapper"])
       + "\ntheir_explain = explain_score\n")
_bde_src = open(f"{P}/v13/frag_up.py").read()
frg += ("\n" + block(_bde_src, "def _regime", ["\ndef _bond_bde"])
        + "\nbde_regime = _regime\n")
adduct_consts = (block(fpf, "ADDUCT_LIST", ["\ndef instr_family"]) + "\n"
                   + block(fpf, "def instr_family", ["\ndef prep_peaks"]))

run = run.replace("import torch\n", "")
run = re.sub(r"^\s*(import|from) v\d+\.\w+.*\n", "", run, flags=re.M)
run = run.replace("import v14.libsearch as L\n", "")
run = run.replace("import v13.fork_fpnet_raw as F\n", "")
run = re.sub(r"\bF\.(FPNet|_MODEL|_logits_raw|fp_and_mass|prep_peaks|_merge_peaks)\b", r"\1", run)
run = re.sub(r"\bL\.(lib_sim|build_rep|analog_sim)\b", r"\1", run)
run = run.replace("L.neutral_mass", "lib_neutral_mass")
run = re.sub(r"\bF\.(FPNet|_MODEL|_logits_raw|fp_and_mass|prep_peaks|_merge_peaks)\b", r"\1", run)
old_paths = (f'PROJECT = "{P}"\n'
             'IN = os.environ.get("CASMI_IN", f"{PROJECT}/data")\n'
             'OUT = os.environ.get("CASMI_OUT", PROJECT)\n'
             'FP = os.environ.get("CASMI_FP", f"{PROJECT}/data")')
new_paths = ('import glob as _glob\n'
             '_comp = _glob.glob("/kaggle/input/**/test.parquet", recursive=True)\n'
             '_fp = _glob.glob("/kaggle/input/**/coconut_fp.parquet", recursive=True)\n'
             'IN = __import__("os").path.dirname(_comp[0]) if _comp else "data"\n'
             'FP = __import__("os").path.dirname(_fp[0]) if _fp else IN\n'
             'OUT = "/kaggle/working" if _comp else "."\n'
             'print("IN=", IN, "FP=", FP, "OUT=", OUT)')
assert old_paths in run, "paths block not found"
run = run.replace(old_paths, new_paths)
for _name in ("ble", "lbs", "rnk", "fpfrag", "frg", "run"):
    s = {"ble": ble, "lbs": lbs, "rnk": rnk, "fpfrag": fpfrag, "frg": frg, "run": run}[_name]
    s = re.sub(r'\nif __name__ == "__main__":\n(?:    .*\n?)+', '\n', s)
    assert "__main__" not in s, _name
    if _name == "ble":
        ble = s
    elif _name == "lbs":
        lbs = s
    elif _name == "rnk":
        rnk = s
    elif _name == "fpfrag":
        fpfrag = s
    elif _name == "frg":
        frg = s
    else:
        run = s + "\nmain()\n"
header = ("import math\nimport numpy as np, pandas as pd, torch, glob, os\n"
          "import torch.nn as nn\nimport torch.nn.functional as F\n"
          "from types import SimpleNamespace\n"
          "HAVE_RDKIT = True\n_g = {}\n_MODEL = None\n"
          + adduct_consts +
          "CFG = SimpleNamespace(PPM_WIN=8.5, MZ_TOL=0.01, INT_FLOOR=0.002,\n"
          "    MAX_PEAKS=256, INT_POWER=1.0, ENT_WEIGHT=True,\n"
          "    ANALOG_WIN=200.0, N_ANALOG=100, SIM_POWER=4.0)\n")
for _name, _s in [("sub", sub), ("ble", ble), ("lbs", lbs), ("rnk", rnk),
                  ("fpfrag", fpfrag), ("frg", frg), ("run", run)]:
    compile(header + _s if _name in ("fpfrag", "lbs", "rnk", "frg", "run") else _s,
            _name, "exec")
full = header + sub + ble + lbs + rnk + fpfrag + frg + run
assert "Users/martin" not in full, "local path leak!"
assert not re.search(r"^\s*(import|from)\s+v\d+\.", full, flags=re.M), "module import leak!"
assert full.count("\nmain()\n") == 1
cells = [
    {"cell_type": "markdown", "metadata": {},
     "source": ["# CASMI26 v14: cleaned floor + their full-stack fills\n",
                "CPU-only, offline, torch+sklearn. Writes `submission.csv`."]},
    *[{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": [line + "\n" for line in s.splitlines()]}
      for s in (header, sub, ble, lbs, rnk, fpfrag, frg, run)],
]
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(f"{P}/kernels/baseline-v0/notebook.ipynb", "w"))
print("v14 kernel notebook OK")
