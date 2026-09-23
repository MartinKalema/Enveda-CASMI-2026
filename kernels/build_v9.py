"""Build kernels/baseline-v0/notebook.ipynb from v9 sources.
Usage: .venv/bin/python kernels/build_v9.py
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
cha = open(f"{P}/v4/channels.py").read()
run = open(f"{P}/v9/submit_v9.py").read()

ble = ble.replace("from v1.subformula import ADDUCT_DELTA\n", "")
ble = ble.replace(f'PROJECT = "{P}"', 'PROJECT = "."  # unused in kernel')
# submit_v9 inlines channels via local aliases; drop versioned imports
run = re.sub(r"^from v\d+\.\w+ import .*\n", "", run, flags=re.M)
run = run.replace("from v4.channels import entropy_similarity as _es\n", "")
run = run.replace("from v4.channels import entropy_similarity as _es2\n", "")
run = run.replace("from v4.channels import tanimoto as _tn\n", "")
run = run.replace("_es(", "entropy_similarity(")
run = run.replace("_es2(", "entropy_similarity(")
run = run.replace("_tn(", "tanimoto(")
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

for _name in ("ble", "cha", "run"):
    s = {"ble": ble, "cha": cha, "run": run}[_name]
    s = re.sub(r'\nif __name__ == "__main__":\n(?:    .*\n?)+', '\n', s)
    assert "__main__" not in s, _name
    if _name == "ble":
        ble = s
    elif _name == "cha":
        cha = s
    else:
        run = s + "\nmain()\n"
assert "def main(" not in ble and "def main(" not in cha

for _name, _s in [("sub", sub), ("ble", ble), ("cha", cha), ("run", run)]:
    compile(_s, _name, "exec")
full = sub + ble + cha + run
assert "Users/martin" not in full, "local path leak!"
assert not re.search(r"^from v\d|^import v\d", full, flags=re.M), "module import leak!"
assert not re.search(r"^\s*(import|from)\s+rdkit", full, flags=re.M), "rdkit import leak!"
assert not re.search(r"^\s*(import|from)\s+torch", full, flags=re.M), "torch import leak!"
assert full.count("\nmain()\n") == 1

cells = [
    {"cell_type": "markdown", "metadata": {},
     "source": ["# CASMI26 v9: cosine floor + GBM-25 fills\n",
                "CPU-only, offline, sklearn. Writes `submission.csv`."]},
    *[{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": [line + "\n" for line in s.splitlines()]} for s in (sub, ble, cha, run)],
]
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(f"{P}/kernels/baseline-v0/notebook.ipynb", "w"))
print("v9 kernel notebook OK")
