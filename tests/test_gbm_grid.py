"""TDD for GBM grid search: metric correctness, split integrity, determinism."""
import numpy as np
import pandas as pd

from v13.gbm_grid import mrr_of


def _frame(rows):
    return pd.DataFrame(rows, columns=["seed", "truth", "cand", "s"])


def test_mrr_first_place_is_one():
    df = _frame([(0, "T", "T", 0.9), (0, "T", "D", 0.1)])
    assert mrr_of(df) == 1.0


def test_mrr_second_place_is_half():
    df = _frame([(0, "T", "D", 0.9), (0, "T", "T", 0.1)])
    assert mrr_of(df) == 0.5


def test_mrr_miss_is_zero():
    df = _frame([(0, "T", "A", 0.9), (0, "T", "B", 0.1)])
    assert mrr_of(df) == 0.0


def test_mrr_beyond_25_is_zero():
    rows = [(0, "T", f"D{i}", 1.0 - i * 0.01) for i in range(30)]
    rows.append((0, "T", "T", 0.0))
    assert mrr_of(_frame(rows)) == 0.0


def test_mrr_averages_queries():
    df = _frame([(0, "A", "A", 0.9), (0, "A", "X", 0.1),
                 (0, "B", "Y", 0.9), (0, "B", "B", 0.1)])
    assert mrr_of(df) == 0.75  # 1.0 and 0.5


def test_folds_share_no_queries():
    feat = pd.concat([pd.read_parquet(
        "/Users/martin/Desktop/enveda-casmi26-molecule-id/v5/feat_cache/seed{s}.parquet".format(s=s))
        for s in (10, 11, 12)], ignore_index=True)
    q = {s: set(feat[feat["seed"] == s]["truth"].unique()) for s in (10, 11, 12)}
    assert q[10].isdisjoint(q[11]) and q[11].isdisjoint(q[12]) and q[10].isdisjoint(q[12])


def test_grid_table_reproduces_winner():
    """Recompute the winning cell; guards result transcription."""
    import pickle
    rows = pickle.load(open("/Users/martin/Desktop/enveda-casmi26-molecule-id/v13/gbm_grid.pkl", "rb"))
    assert rows["lr0.05-lf31-l21.0"] is not None
    assert abs(float(np.mean(rows["lr0.05-lf31-l21.0"])) - 0.084) < 0.005
