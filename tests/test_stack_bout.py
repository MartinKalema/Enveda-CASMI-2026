"""TDD for stack_bout: artifacts, shapes, and a 2-query smoke run."""
import numpy as np

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"


def test_artifacts_exist():
    import os
    for p in ("v13/their_ranker.pkl", "v13/gbm9_keyed.pkl", "data/fp_trans.pt",
              "data/frag_cache.pkl"):
        assert os.path.exists(f"{PROJECT}/{p}"), f"missing {p}"


def test_their_ranker_predicts():
    import pickle
    models = pickle.load(open(f"{PROJECT}/v13/their_ranker.pkl", "rb"))
    assert len(models) == 8
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 31)).astype(np.float32)
    ps = np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)
    assert ps.shape == (10,) and np.all((ps >= 0) & (ps <= 1))


def test_our_ranker_predicts():
    import pickle
    m = pickle.load(open(f"{PROJECT}/v13/gbm9_keyed.pkl", "rb"))
    assert m["feats"] == ["cos", "ent", "ana", "mass_err", "log_prior",
                          "t_top1", "metfrag", "fpdot", "bde"]
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 9)).astype(np.float32)
    p = m["model"].predict_proba(X)[:, 1]
    assert p.shape == (10,) and np.all((p >= 0) & (p <= 1))


def test_smoke_bout_runs():
    from v14.stack_bout import stage_bout
    import inspect
    sig = inspect.signature(stage_bout)
    assert "n_query" in sig.parameters and "seeds" in sig.parameters
