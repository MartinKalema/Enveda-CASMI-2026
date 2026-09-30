"""Regenerate v13/fork_fpnet_raw.py from the fork notebook. Single source of
truth for the harness - hand-editing the generated file is banned (TDD rule).
Usage: .venv/bin/python kernels/extract_fpnet.py
"""
import json

P = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
nb = json.load(open(f"{P}/kernels/fork-034/notebook.ipynb"))
full = ""
for c in nb["cells"]:
    if c["cell_type"] == "code":
        full += "".join(c["source"]) + "\n"


def block(s, start, stops):
    i = s.index(start)
    mm = []
    import re
    for st in stops:
        for m in re.finditer(st, s[i + 10:]):
            mm.append(m.start())
    return s[i:i + 10 + min(mm)]


consts = block(full, "ADDUCT_LIST", ["\ndef instr_family"])
consts += block(full, "def instr_family", ["\ndef prep_peaks"])
consts += block(full, "MAX_PEAKS_NN", ["\nADDUCT_LIST"])
parts = [
    "import math",
    "import numpy as np, pandas as pd, torch, glob, os",
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "from rdkit import Chem",
    "from rdkit.Chem import rdFingerprintGenerator",
    "from rdkit.Chem import Descriptors, MACCSkeys",
    "from rdkit.Chem.Descriptors import ExactMolWt",
    "HAVE_RDKIT = True",
    "_g = {}",
    "_MODEL = None",
    consts,
    block(full, "class SinEmb", ["\nclass Block"]),
    block(full, "class Block", ["\nclass FPNet"]),
    block(full, "class FPNet", ["\ndef _merge_peaks"]),
    block(full, "def prep_peaks", ["\nclass SinEmb"]),
    block(full, "def _merge_peaks", ["\ndef model_logits"]),
    block(full, "def model_logits", ["\ndef _logits_from"]),
    block(full, "def _logits_from", ["\ndef _logits_raw"]),
    block(full, "def _logits_raw", ["\n# ==="]),
    block(full, "def _fp_init", ["\ndef fp_and_mass"]),
    block(full, "def fp_and_mass", ["\nclass CandidatePool"]),
]
out = "\n".join(parts)
open(f"{P}/v13/fork_fpnet_raw.py", "w").write(out)
print("regenerated", len(out), "chars")
