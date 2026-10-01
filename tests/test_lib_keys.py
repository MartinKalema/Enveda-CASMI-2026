"""Key-space contract: lib_sim/build_rep keys must be candidate SMILES.
Regression test: self-match of a library spectrum must hit its own SMILES."""
import numpy as np
import pandas as pd

from v14.build_rank_train_bde import nm


def _tiny_lib():
    from v14.loaders import load_library_df
    tr = pd.read_parquet(
        "/Users/martin/Desktop/enveda-casmi26-molecule-id/data/train.parquet",
        columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                 "ms2_mzs", "ms2_normalized_intensities"])
    tr = tr[tr["adduct"] == "[M+H]+"].head(20)
    return tr


def test_lib_sim_keyed_by_smiles():
    import v14.libsearch as L
    from v14.loaders import load_library_df
    tr = _tiny_lib()
    lib = load_library_df(tr, L.neutral_mass)
    r = tr.iloc[0]
    target = float(nm(r["precursor_mz"], r["adduct"]))
    hits = L.lib_sim(lib, [(r["ms2_mzs"], r["ms2_normalized_intensities"], r["adduct"])],
                     target)
    assert r["normalized_smiles"] in hits, \
        f"self-match must key by SMILES, got keys {list(hits)[:3]}"
    assert hits[r["normalized_smiles"]] > 0.9, "self-match must score ~1"


def test_build_rep_keyed_by_smiles():
    import v14.libsearch as L
    from v14.loaders import load_library_df
    tr = _tiny_lib()
    lib = load_library_df(tr, L.neutral_mass)
    rep, rep_key, _, _ = L.build_rep(lib)
    assert set(rep_key.tolist()) <= set(tr["normalized_smiles"].tolist()), \
        "rep keys must be SMILES, not inchikeys"
