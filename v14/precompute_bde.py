"""Precompute OUR BDE frag masses for the production candidate universe.
One entry per (regime): pos [M+H]+, neg [M-H]-, na [M+Na]+.
Universe = same train+COCONUT mass windows as precompute_prod.py."""
import pickle

from v14.precompute_prod import window, nm

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
REGIMES = {"pos": "[M+H]+", "neg": "[M-H]-", "na": "[M+Na]+"}


def _one(args):
    import numpy as np
    from v13.frag_up import fragment_masses_bde
    s, adduct = args
    try:
        return s, np.asarray(fragment_masses_bde(s, adduct), dtype=float)
    except Exception:
        return s, np.zeros(0)


def main():
    import numpy as np
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    import itertools

    test = pd.read_parquet(f"{PROJECT}/data/test.parquet")
    test["neutral"] = [nm(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()
    train = pd.read_parquet(f"{PROJECT}/data/train.parquet",
                            columns=["normalized_smiles", "precursor_mz", "adduct"])
    train["neutral"] = [nm(p, a) for p, a in zip(train["precursor_mz"], train["adduct"])]
    tstruct = train.groupby("normalized_smiles")["neutral"].median()
    tmass = tstruct.sort_values().values
    tsmi = tstruct.sort_values().index.values
    cf = pd.read_parquet(f"{PROJECT}/data/coconut_fp.parquet",
                         columns=["canonical_smiles", "exact_molecular_weight"])
    co = cf.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values
    uni = set()
    for mol in mol_neutral.index:
        qmass = float(mol_neutral.loc[mol])
        uni.update(window(tmass, tsmi, qmass))
        uni.update(window(cmass, csmi, qmass, cap=3000))
    uni = sorted(uni)
    print(f"universe: {len(uni)}", flush=True)
    for reg, adduct in REGIMES.items():
        jobs = [(s, adduct) for s in uni]
        out = {}
        with ProcessPoolExecutor(max_workers=8) as ex:
            for i, (s, f) in enumerate(ex.map(_one, jobs, chunksize=50)):
                out[s] = f
                if (i + 1) % 40000 == 0:
                    print(f"{reg} {i + 1}/{len(uni)}", flush=True)
        with open(f"{PROJECT}/data/bde_frag_{reg}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"saved {reg}: {len(out)}", flush=True)


if __name__ == "__main__":
    main()
