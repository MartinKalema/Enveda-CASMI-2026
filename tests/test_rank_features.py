"""TDD for the rank_features replica: shape, views, agreement block.
Copied logic, not weights - every property asserted before use.
"""
import numpy as np


def test_rank_features_shape():
    from v14.rank import rank_features
    rng = np.random.default_rng(0)
    nc, na, nb = 50, 10, 6930
    X = rank_features(rng.integers(0, 2, (nc, nb)).astype(np.float32),
                      rng.random(nc).astype(np.float32),
                      rng.integers(0, 2, (na, nb)).astype(np.float32),
                      np.sort(rng.random(na).astype(np.float32))[::-1],
                      model_logits=rng.normal(size=nb).astype(np.float32),
                      frag=rng.random(nc).astype(np.float32))
    assert X.shape == (nc, 31), X.shape
    assert np.all(np.isfinite(X))


def test_rank_features_nones():
    from v14.rank import rank_features
    rng = np.random.default_rng(1)
    X = rank_features(rng.integers(0, 2, (20, 100)).astype(np.float32),
                      rng.random(20).astype(np.float32),
                      None, np.array([]),
                      model_logits=None, frag=None)
    assert X.shape == (20, 31)
    assert np.all(np.isfinite(X))


def test_agreement_block_logic():
    """When library's best is also model's best, agree==1."""
    from v14.rank import rank_features
    nc, nb = 5, 8
    cf = np.zeros((nc, nb), np.float32)
    cf[2, :] = 1.0
    lv = np.array([0.1, 0.2, 0.9, 0.3, 0.1], np.float32)
    z = np.ones(nb, np.float32)
    X = rank_features(cf, lv, None, np.array([]), model_logits=z, frag=None)
    agree_col = X[:, 27]
    assert (agree_col == 1.0).all(), agree_col
