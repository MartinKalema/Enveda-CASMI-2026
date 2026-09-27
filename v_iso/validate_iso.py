"""Isotope agreement MRR on v5 cached queries (150, 3 seeds)."""
import numpy as np, pandas as pd
from v_iso.iso_features import observed_satellites, iso_scores

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
SEEDS = (10, 11, 12)


def main():
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet",
        columns=["normalized_smiles", "molecular_formula", "adduct", "ms2_mzs", "ms2_normalized_intensities"])
    fmap = dict(zip(tr["normalized_smiles"], tr["molecular_formula"]))
    tr = tr.set_index("normalized_smiles", append=False)
    cf = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet", columns=["canonical_smiles", "molecular_formula"])
    for s, f in zip(cf["canonical_smiles"], cf["molecular_formula"]):
        fmap.setdefault(s, f)
    out = []
    for seed in SEEDS:
        d = pd.read_parquet(f"{PROJECT}/v5/feat_cache/seed{seed}.parquet")
        for (sd, truth), g in d.groupby(["seed", "truth"]):
            try:
                rows = tr.loc[[truth]].head(3)
            except KeyError:
                continue
            if len(rows) == 0:
                continue
            o1, o2 = observed_satellites([(r["ms2_mzs"], r["ms2_normalized_intensities"])
                                          for _, r in rows.iterrows()])
            sc = {}
            for _, c in g.iterrows():
                a, b, h = iso_scores(fmap.get(c["cand"], ""), o1, o2)
                sc[c["cand"]] = a + b + h
            rnk = sorted(sc.items(), key=lambda kv: -kv[1])
            rank = next((i + 1 for i, (s, _) in enumerate(rnk) if s == truth), 10**9)
            out.append(1 / rank if rank <= 25 else 0.0)
    print(f"isotope-agreement MRR25={np.mean(out):.3f} (n={len(out)})")


if __name__ == "__main__":
    main()
