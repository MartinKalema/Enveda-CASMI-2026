# v8: fingerprint transformer with hard in-window decoys (in progress)

Plain English: our two fingerprint attempts failed for known reasons (no hard
negatives, weak MLP, wrong checkpoint). The survey's recipe that scores 0.468
alone: spectrum encoder trained with 63 same-mass decoys under softmax loss,
ranked by raw-logit dot-product, best-validation checkpoint, peak-dropout /
intensity-jitter / mz-noise augmentation. This is the engine's core channel
and the only documented path from 0.095 toward 0.2+.
