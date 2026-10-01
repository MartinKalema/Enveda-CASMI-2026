"""Behavioral contracts for the original-COCONUT pool builders.
Includes the test that would have caught the RDKit pickling bug:
worker functions must survive a real 2-worker pool."""
import pickle

import numpy as np

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def test_pool_parquet_contract():
    import pandas as pd
    df = pd.read_parquet(f"{PROJECT}/data/coconut_orig.parquet")
    assert len(df) > 700000, "must carry the full Oct-2026 release"
    assert {"canonical_smiles", "exact_molecular_weight"}.issubset(df.columns)
    assert int(df["canonical_smiles"].isnull().sum()) == 0
    assert int(df["canonical_smiles"].duplicated().sum()) == 0
    assert bool(np.isfinite(df["exact_molecular_weight"].values).all())


def test_universe_covers_submit_windows():
    """Every SMILES the submit windows can emit must be in the universe."""
    import pandas as pd
    from v14.build_coconut_orig import window10
    from v14.precompute_prod import nm
    uni = set(pickle.load(open(f"{PROJECT}/data/orig_universe.pkl", "rb")))
    test = pd.read_parquet(f"{PROJECT}/data/test.parquet")
    test["neutral"] = [nm(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])]
    mol_neutral = test.groupby("molecule_id")["neutral"].median()
    co = pd.read_parquet(f"{PROJECT}/data/coconut_orig.parquet")
    co = co.sort_values("exact_molecular_weight").reset_index(drop=True)
    cmass = co["exact_molecular_weight"].values
    csmi = co["canonical_smiles"].values
    missing = 0
    total = 0
    for mol in mol_neutral.index:
        for s in window10(cmass, csmi, float(mol_neutral.loc[mol])):
            total += 1
            if s not in uni:
                missing += 1
    assert missing == 0, f"{missing}/{total} window SMILES outside universe"


def test_fp_worker_multiprocessing():
    """RDKit generators must be built inside workers (pickling regression)."""
    from concurrent.futures import ProcessPoolExecutor
    from v14.build_orig_artifacts import _fp_one, _fp_init
    with ProcessPoolExecutor(max_workers=2, initializer=_fp_init) as ex:
        out = list(ex.map(_fp_one, ["CCO", "c1ccccc1"], chunksize=1))
    assert len(out) == 2
    for s, v in out:
        assert v is not None and len(v) == 6930, f"bad fp for {s}"


def test_fp_packed_matches_universe():
    uni = pickle.load(open(f"{PROJECT}/data/orig_fp_keys.pkl", "rb"))
    M = np.load(f"{PROJECT}/data/orig_fp_packed.npy", mmap_mode="r")
    assert len(uni) == M.shape[0] and M.shape[1] == 867
    assert len(set(uni)) == len(uni), "keys must be unique and ordered"
    ref = pickle.load(open(f"{PROJECT}/data/orig_universe.pkl", "rb"))
    assert uni == ref, "packed keys must equal universe order"


def test_bde_covers_universe_per_regime():
    uni = set(pickle.load(open(f"{PROJECT}/data/orig_universe.pkl", "rb")))
    for reg in ("pos", "neg", "na"):
        d = pickle.load(open(f"{PROJECT}/data/orig_bde_{reg}.pkl", "rb"))
        assert uni.issubset(set(d)), f"{reg} missing SMILES"


def test_reuse_equals_fresh_compute():
    """Reused cache rows must equal freshly computed values."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
    from v13.frag_up import fragment_masses_bde
    uni = pickle.load(open(f"{PROJECT}/data/orig_universe.pkl", "rb"))
    old_keys = pickle.load(open(f"{PROJECT}/data/their_fp_keys.pkl", "rb"))
    overlap = [s for s in uni if s in set(old_keys)]
    assert overlap, "expected overlap with previous universe"
    s = overlap[0]
    M = np.load(f"{PROJECT}/data/orig_fp_packed.npy", mmap_mode="r")
    i = uni.index(s)
    bits = np.load("/tmp/cocofp/fp_bits.npy")
    m = Chem.MolFromSmiles(s)
    m2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    m3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    rk = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
    v = np.concatenate([m2.GetFingerprintAsNumPy(m).astype(np.uint8),
                        m3.GetFingerprintAsNumPy(m).astype(np.uint8),
                        rk.GetFingerprintAsNumPy(m).astype(np.uint8),
                        np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[bits]
    assert (np.unpackbits(M[i])[:6930] == v).all(), "packed row != fresh fp"
    d = pickle.load(open(f"{PROJECT}/data/orig_bde_pos.pkl", "rb"))
    assert np.allclose(np.asarray(d[s], float),
                       np.asarray(fragment_masses_bde(s, "[M+H]+"), float)), \
        "reused BDE != fresh BDE"
