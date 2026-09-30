"""TDD for extracted library search: identical spectrum must self-match."""
import numpy as np


def test_lib_sim_self_match():
    import v14.libsearch as L
    import pandas as pd
    PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet",
                         columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                                  "ms2_mzs", "ms2_normalized_intensities"]).head(200)
    from v14.loaders import load_library_df
    lib = load_library_df(tr, L.neutral_mass)
    q = tr.iloc[0]
    from v14.libsearch import neutral_mass as _nm
    target = float(_nm(np.array([q["precursor_mz"]]), np.array([q["adduct"]]))[0])
    hits = L.lib_sim(lib, [(q["ms2_mzs"], q["ms2_normalized_intensities"], q["adduct"])],
                     target)
    assert hits, "self-match found nothing"
    assert max(hits.values()) > 0.5, f"self-match too weak: {max(hits.values())}"


def test_analog_sim_self_present():
    import v14.libsearch as L
    import pandas as pd
    PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
    tr = pd.read_parquet(f"{PROJECT}/data/train.parquet",
                         columns=["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
                                  "ms2_mzs", "ms2_normalized_intensities"]).head(200)
    from v14.loaders import load_library_df
    lib = load_library_df(tr, L.neutral_mass)
    rep, rep_key, rep_nm, rep_ad = L.build_rep(lib)
    assert len(rep) > 0
    q = tr.iloc[0]
    from v14.libsearch import neutral_mass as _nm2
    target2 = float(_nm2(np.array([q["precursor_mz"]]), np.array([q["adduct"]]))[0])
    out = L.analog_sim(lib, [(q["ms2_mzs"], q["ms2_normalized_intensities"], q["adduct"])],
                       target2, rep, rep_key, rep_nm, rep_ad)
    assert len(out) > 0
