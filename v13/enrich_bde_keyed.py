"""Keyed BDE enrichment: (seed, truth, cand, bde). Joins must be by key,
never by position (lesson: positional merge of groupby-sorted output
against unsorted cache silently shuffles features into garbage).
"""
import numpy as np
import pandas as pd

from v13.frag_up import cached_fragments_bde
from v13.enrich_bde import ladder_score

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)


def main():
    tr = pd.read_parquet(
        f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "adduct", "ms2_mzs", "ms2_normalized_intensities"])
    tr = tr.set_index("normalized_smiles", append=False)
    out = []
    for seed in SEEDS:
        d = pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{seed}.parquet")
        rows = []
        for (sd, truth), g in d.groupby(["seed", "truth"]):
            try:
                qrows = tr.loc[[truth]].head(2)
            except KeyError:
                qrows = None
            if qrows is None or len(qrows) == 0:
                for _, c in g.iterrows():
                    rows.append((sd, truth, c["cand"], 0.0))
                continue
            for _, c in g.iterrows():
                best = 0.0
                for _, r in qrows.iterrows():
                    f = cached_fragments_bde(c["cand"], r["adduct"])
                    v = ladder_score(r["ms2_mzs"], r["ms2_normalized_intensities"], f)
                    if v > best:
                        best = v
                rows.append((sd, truth, c["cand"], best))
        q = pd.DataFrame(rows, columns=["seed", "truth", "cand", "bde"])
        q.to_parquet(f"{PROJECT}/v13/feat_bde_keyed{seed}.parquet")
        print(f"seed {seed}: {len(q)} rows", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
