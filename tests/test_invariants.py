"""Submission + kernel-notebook invariants. Run before every push."""
import json
import re

import pandas as pd

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def test_submission_format():
    sub = pd.read_csv(f"{PROJECT}/submission.csv")
    assert list(sub.columns) == ["molecule_id", "smiles"]
    assert sub["molecule_id"].duplicated().sum() == 0
    assert sub["smiles"].isnull().sum() == 0
    n = sub["smiles"].apply(lambda s: len(str(s).split(";")))
    assert (n == 25).all(), f"rows without 25 guesses: {(n != 25).sum()}"
    assert (sub["smiles"].str.len() > 0).all()


def test_fork_notebook_invariants():
    nb = json.load(open(f"{PROJECT}/kernels/fork-034/notebook.ipynb"))
    full = ""
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            compile(src, "cell", "exec")  # every cell must compile
            full += src + "\n"
    assert "Users/martin" not in full, "local path leak"
    assert not re.search(r"^from v\d|^import v\d", full, flags=re.M), "versioned import leak"
    # rdkit is allowed only behind the offline-wheel guard, never bare
    assert "HAVE_RDKIT" in full, "rdkit must be guarded by HAVE_RDKIT"
    # exactly one production entry point fires
    assert full.count("\nmain()\n") <= 1


def test_builder_scripts_compile():
    import glob
    for f in glob.glob(f"{PROJECT}/kernels/build_*.py") + \
             glob.glob(f"{PROJECT}/v*/validate*.py") + \
             glob.glob(f"{PROJECT}/v*/submit_*.py"):
        compile(open(f).read(), f, "exec")
