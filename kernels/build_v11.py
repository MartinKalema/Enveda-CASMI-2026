"""Build kernels/baseline-v0/notebook.ipynb from v11 sources.
Usage: .venv/bin/python kernels/build_v11.py
"""
import json
import re

P = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def block(s, start, stops):
    i = s.index(start)
    j = len(s)
    for st in stops:
        k = s.find(st, i + 1)
        if k != -1:
            j = min(j, k)
    return s[i:j]


sub = open(f"{P}/v1/subformula.py").read()
ble = open(f"{P}/v2/blend.py").read()
tsrc = open(f"{P}/v2/train_fp.py").read()
f7 = open(f"{P}/v7/submit_v7.py").read()
cha = open(f"{P}/v4/channels.py").read()
run = open(f"{P}/v11/submit_v11.py").read()

ble = ble.replace("from v1.subformula import ADDUCT_DELTA\n", "")
ble = ble.replace(f'PROJECT = "{P}"', 'PROJECT = "."  # unused in kernel')

mlp = ("import torch\nimport torch.nn as nn\n"
       + block(tsrc, "BIN_W, MZ_MAX", ["def meta_vec"])
       + block(tsrc, "ADDUCTS = [", ["def meta_vec"])
       + block(tsrc, "def bin_spectrum", ["ADDUCTS = ["])
       + block(tsrc, "def meta_vec", ["class FpMLP"])
       + block(tsrc, "class FpMLP", ["def load_frame"]))

frag = ("import pickle as _pk\n"
        + block(f7, 'AD = {"[M+H]+', ["def neutral_mass"])
        + block(f7, "def frag_match", ["def main("]))

run = run.replace("import torch\n", "")
run = re.sub(r"^from v\d+\.\w+ import .*\n", "", run, flags=re.M)
run = run.replace("from v7.submit_v7 import frag_match\n", "")
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

for _name in ("ble", "cha", "frag", "run"):
    s = {"ble": ble, "cha": cha, "frag": frag, "run": run}[_name]
    s = re.sub(r'\nif __name__ == "__main__":\n(?:    .*\n?)+', '\n', s)
    assert "__main__" not in s, _name
    if _name == "ble":
        ble = s
    elif _name == "cha":
        cha = s
    elif _name == "frag":
        frag = s
    else:
        run = s + "\nmain()\n"
assert "def main(" not in mlp and "def main(" not in frag
assert "def load_frame" not in mlp

for _name, _s in [("sub", sub), ("ble", ble), ("mlp", mlp), ("frag", frag),
                  ("cha", cha), ("run", run)]:
    compile(_s, _name, "exec")
full = sub + ble + mlp + frag + cha + run
assert "Users/martin" not in full, "local path leak!"
assert not re.search(r"^from v\d|^import v\d", full, flags=re.M), "module import leak!"
assert not re.search(r"^\s*(import|from)\s+rdkit", full, flags=re.M), "rdkit import leak!"
assert full.count("\nmain()\n") == 1

cells = [
    {"cell_type": "markdown", "metadata": {},
     "source": ["# CASMI26 v11: cleaned floor + GBM-8 fills\n",
                "CPU-only, offline, torch+sklearn. Writes `submission.csv`."]},
    *[{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": [line + "\n" for line in s.splitlines()]}
      for s in (sub, ble, mlp, frag, cha, run)],
]
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(f"{P}/kernels/baseline-v0/notebook.ipynb", "w"))
print("v11 kernel notebook OK")
