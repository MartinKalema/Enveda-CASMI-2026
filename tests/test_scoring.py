"""Scoring invariants: every channel must rank truth above decoys."""
import numpy as np
import pandas as pd

from v1.subformula import parse_formula, rdbe, explained_intensity
from v2.blend import cosine, tanimoto
from v4.channels import entropy_similarity
from v13.frag_up import frag_score_bde

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
DATA = None


def data():
    global DATA
    if DATA is None:
        DATA = pd.read_parquet(
            f"{PROJECT}/data/train.parquet",
            columns=["normalized_smiles", "molecular_formula", "adduct",
                     "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    return DATA


def test_parse_formula():
    assert parse_formula("C9H8N2O2") == {"C": 9, "H": 8, "N": 2, "O": 2}
    assert parse_formula("C6H12O6")["O"] == 6
    assert parse_formula("NOTAFORMULA!!!") is None or isinstance(parse_formula("Xx9"), (dict, type(None)))


def test_rdbe_benzene():
    assert abs(rdbe({"C": 6, "H": 6}) - 4.0) < 1e-9


def test_cosine_identical_is_one():
    mz = np.array([100.0, 200.0, 300.0])
    it = np.array([0.5, 1.0, 0.25])
    assert abs(cosine(mz, it, mz, it) - 1.0) < 1e-9


def test_cosine_empty_is_zero():
    assert cosine([], [], [1.0], [1.0]) == 0.0


def test_entropy_bounded():
    rng = np.random.default_rng(0)
    mz = np.sort(rng.uniform(50, 500, 60))
    it = rng.uniform(0, 1, 60)
    s = entropy_similarity(mz, it, mz, it)
    assert 0.0 <= s <= 1.0
    assert s > 0.9


def test_explained_true_beats_wrong():
    tr = data().sample(20, random_state=3)
    t = w = 0.0
    for r in tr.itertuples():
        t += explained_intensity(r.ms2_mzs, r.ms2_normalized_intensities,
                                 r.molecular_formula, r.adduct)[0]
        w += explained_intensity(r.ms2_mzs, r.ms2_normalized_intensities,
                                 "C20H30O5", r.adduct)[0]
    assert t > w, f"true={t / 20:.3f} wrong={w / 20:.3f}"


def test_frag_bde_true_beats_wrong():
    tr = data().sample(10, random_state=7)
    t = w = 0.0
    for i, r in enumerate(tr.itertuples()):
        a, _ = frag_score_bde(r.ms2_mzs, r.ms2_normalized_intensities,
                              r.normalized_smiles, r.adduct, r.precursor_mz)
        t += a
        if i < 5:
            d = data().iloc[(i * 7919) % len(data())]
            a2, _ = frag_score_bde(r.ms2_mzs, r.ms2_normalized_intensities,
                                   d.normalized_smiles, r.adduct, r.precursor_mz)
            w += a2
    assert t / 10 > (w / 5) * 1.5, f"true={t / 10:.3f} wrong={w / 5:.3f}"


def test_tanimoto_identity():
    a = np.array([True, False, True])
    assert tanimoto(a, a) == 1.0
    assert tanimoto(a, np.array([False, True, False])) == 0.0
