"""Train THEIR ranker config on THEIR rank_train.npz (8 models).
Provenance: prvsiyan/casmi26-ranker-features (public dataset).
Saves v13/their_ranker.pkl
"""
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
W1_PRIORS = (0.30, 0.60)
SEEDS = (0, 1, 2, 3)


def main():
    z = np.load("/tmp/ranker/rank_train.npz")
    X, Y, M = z["X"], z["Y"], z["M"]
    print("X", X.shape, "pos rate", Y.mean(), flush=True)
    models = {}
    for w1 in W1_PRIORS:
        W = np.where(M == 0, w1, 1.0 - w1)
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(
                max_depth=6, max_iter=500, learning_rate=0.03,
                min_samples_leaf=80, l2_regularization=1.0, random_state=sd)
            m.fit(X, Y, sample_weight=W)
            models[(w1, sd)] = m
            print(f"w1={w1} seed={sd} done", flush=True)
    pickle.dump(models, open(f"{PROJECT}/v13/their_ranker.pkl", "wb"))
    print("saved v13/their_ranker.pkl", flush=True)


if __name__ == "__main__":
    main()
