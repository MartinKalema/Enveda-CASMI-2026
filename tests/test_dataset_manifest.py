"""Dataset-manifest gate: every FP-dir file the production script loads must
exist in the kernel dataset dir. Catches local-only artifacts before push.
(Kaggle-side readiness is checked at push time via `datasets files`.)"""
import os
import re

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
DS = f"{PROJECT}/datasets/coconut-fp"
# competition inputs, not our dataset
NOT_Ours = {"test.parquet", "train.parquet"}


def _needed():
    s = open(f"{PROJECT}/v14/submit_v14.py").read()
    out = set()
    for m in re.finditer(r'\{FP\}/([^"\']+)', s):
        pat = m.group(1)
        if "{_reg}" in pat:
            out.update(pat.replace("{_reg}", r) for r in ("pos", "neg", "na"))
        else:
            out.add(pat)
    b = open(f"{PROJECT}/kernels/build_v14.py").read()
    out.update(re.findall(r'glob\("/kaggle/input/\*\*/([A-Za-z0-9_.-]+)"', b))
    return {f for f in out if f not in NOT_Ours}


def test_manifest_complete():
    have = set(os.listdir(DS))
    missing = {f for f in _needed() if f not in have}
    assert not missing, f"dataset dir missing: {sorted(missing)}"
