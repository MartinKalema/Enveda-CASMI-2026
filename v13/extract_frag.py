"""Extract the fork's frag channel into an importable module for head-to-head tests."""
import json
import re

P = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
nb = json.load(open(f"{P}/kernels/fork-034/notebook.ipynb"))
full = ""
for c in nb["cells"]:
    if c["cell_type"] == "code":
        full += "".join(c["source"]) + "\n"

# CFG constants used by frag channel
cfg = re.search(r"INT_FLOOR\s*=.*\nMAX_PEAKS\s*=.*\nINT_POWER\s*=.*\nENT_WEIGHT\s*=.*\nMZ_TOL\s*=.*\n", full)
print("CFG block found:", bool(cfg))

# function/class blocks to extract (up to next top-level def/class or section banner)
names = ["AMU =", "H_ATOM =", "HAVE_RDKIT", "def _clean", "def mol_graph",
         "def _components", "def _frag_masses_wrapper", "def explain_score",
         "def frag_scores"]
for n in names:
    print(n, "->", "FOUND" if n in full else "MISSING")
