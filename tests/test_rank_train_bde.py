"""Builder contract: BDE-based rank_train must mirror THEIR npz format.
Same 31 cols, Y labels, M class flags, G groups — only frag source differs.
Feature code path must be OUR rank_features (the same fn inference uses)."""
import numpy as np

NPZ = "/tmp/ranker/rank_train.npz"
OUT = "/Users/martin/Desktop/enveda-casmi26-molecule-id/data/rank_train_bde.npz"
BUILDER = "/Users/martin/Desktop/enveda-casmi26-molecule-id/v14/build_rank_train_bde.py"


def test_their_format():
    z = np.load(NPZ)
    assert z["X"].shape[1] == 31
    assert set(np.unique(z["M"])).issubset({0.0, 1.0})
    assert z["X"].shape[0] == z["Y"].shape[0] == z["M"].shape[0] == z["G"].shape[0]


def test_builder_uses_same_feature_code():
    s = open(BUILDER).read()
    assert "from v14.rank import rank_features" in s, \
        "retrain must use the same rank_features fn as inference"
    assert "cached_fragments_bde" in s or "bde_frag_" in s, \
        "frag features must come from BDE masses"
    assert "_frag_masses_wrapper" not in s and "their_frag_uni" not in s, \
        "their frag masses must not leak into retrain"


def test_builder_leaves_query_spectrum_out():
    s = open(BUILDER).read()
    assert "q.name" in s and "drop" in s, \
        "class-1 library must exclude the query spectrum itself (else lv~1 giveaway)"


def test_bde_matrix_mirrors():
    z0 = np.load(NPZ)
    z1 = np.load(OUT)
    assert z1["X"].shape[1] == 31
    n = z1["X"].shape[0]
    assert z1["Y"].shape[0] == z1["M"].shape[0] == z1["G"].shape[0] == n
    assert set(np.unique(z1["Y"])).issubset({0.0, 1.0})
    assert set(np.unique(z1["M"])).issubset({0.0, 1.0})
    pr = z1["Y"].mean()
    assert 0.001 < pr < 0.05, f"pos rate {pr} off vs theirs {z0['Y'].mean():.4f}"
    assert len(np.unique(z1["G"])) > 100, "need query groups, not one blob"
