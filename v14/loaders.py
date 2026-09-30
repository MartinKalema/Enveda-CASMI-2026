"""Dataframe adapter for the extracted library: same L dict, no parquet needed."""
import numpy as np


def load_library_df(df, neutral_mass):
    """df columns: inchikey14, normalized_smiles, adduct, precursor_mz,
    ms2_mzs, ms2_normalized_intensities. Returns L dict like load_library."""
    mz_list = [np.asarray(m, dtype=np.float32) for m in df["ms2_mzs"]]
    it_list = [np.asarray(v, dtype=np.float32) for v in df["ms2_normalized_intensities"]]
    off = np.zeros(len(df) + 1, np.int64)
    for i, m in enumerate(mz_list):
        off[i + 1] = off[i] + len(m)
    allmz = np.concatenate(mz_list) if mz_list else np.zeros(0, np.float32)
    allin = np.concatenate(it_list) if it_list else np.zeros(0, np.float32)
    prec = np.asarray(df["precursor_mz"], dtype=np.float64)
    add = np.asarray(df["adduct"].astype(str).tolist(), dtype=object)
    ik = np.asarray(df["inchikey14"].astype(str).tolist(), dtype=object)
    smi = np.asarray(df["normalized_smiles"].astype(str).tolist(), dtype=object)
    nm = neutral_mass(prec, add)
    ok = np.isfinite(nm)
    order = np.argsort(np.where(ok, nm, 1e18), kind="mergesort")
    best = {}
    for k, s in zip(ik, smi):
        if k and s and k not in best:
            best[k] = s
    return dict(off=off, mz=allmz, it=allin, nm=nm, ik=ik, best=best,
                order=order, snm=nm[order], n_ok=int(ok.sum()), ad=add)
