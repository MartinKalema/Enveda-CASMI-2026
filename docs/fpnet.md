# fpnet: set-transformer fingerprint network (v-fpnet)

Fingerprint network done RIGHT per finding 5 of `docs/kaggle-notebooks-survey.md`:
rank by raw-logit dot product `f·z`, train with hard in-window decoys under
softmax CE, augment, keep the best-by-retrieval checkpoint. The one deliberate
difference from v2/v8: a **set-transformer over subformula-annotated peaks**
instead of a raw binned-spectrum MLP.

## Model

- Input: top-80 peaks as a set. Per peak: mz, intensity, sqrt-intensity,
  neutral loss, mass defect, **subformula annotation** (peak mass neutralised
  by the precursor adduct delta, matched against 28,939 plausible subformula
  masses from the 150 most frequent training formulae; hit flag + log-ppm —
  truth-free, works at test time), rank fraction.
- Covariates: adduct (top-15 + other), instrument bucket, polarity, precursor
  mz, CE mean/count/flag, peak count (31 dims).
- Peak MLP → 2 custom pre-norm encoder layers (d=256, 4 heads, additive pad
  masks — MPS has no nested-tensor fast path, so no `nn.TransformerEncoder`)
  → single-query attention pooling → concat covariates → 1024 MLP → 2048 raw
  logits. Ranking is raw `f·z` (f = 0/1 Morgan bits).

## Training

- Softmax CE over truth + 63 same-mass decoys from `max(10ppm, 0.01 Da)`
  window, fresh decoys each epoch, logits `(F·z)/10`. AdamW 2e-4 (wd 1e-4),
  cosine schedule. Phase 1: 16 epochs from scratch; phase 2: warm restart
  from best + 18 epochs (34 total), 150k spectra.
- Augmentation: peak dropout p=0.2 (top-4 kept), intensity jitter U(0.8,1.2),
  mz noise 5 ppm.
- Split: inchikey14 95/5 structure-disjoint. Pooled queries: held-out
  *spectra* of pool structures (truth in pool, memorization-prone).
  Disjoint queries: val-split structures = novel chemistry, truth injected
  into the candidate window (simulates test).

## Results (800 queries each, raw f·z ranking, identical protocol)

| model | pooled @1/@5/@10 | disjoint @1/@5/@10 |
|---|---|---|
| v2 BCE MLP | 0.100 / 0.269 / 0.351 | 0.086 / 0.278 / 0.360 |
| v8 decoy MLP (binned) | 0.376 / 0.670 / 0.756 | 0.335 / 0.630 / 0.730 |
| **v-fpnet set-transformer** | **0.303 / 0.599 / 0.683** | **0.254 / 0.548 / 0.659** |

(fpnet best = phase-2 ep10 on pooled@5; disjoint reported at that checkpoint.
MLP baselines from `v-fpnet/baseline_eval.py`; v8's 0.80 was cosine ranking,
raw-dot pooled is 0.670 — ranking geometry matters.)

## Honest takeaway — target missed, and why

- fpnet beats v2 2.2× (0.269 → 0.599 pooled@5) and beats the old 0.038-class
  disjoint baseline far out of sight (disjoint@5 0.548), but it does **not**
  beat the v8 binned MLP (0.670 / 0.630) and is far from 0.80 pooled.
- Phase-2 loss kept falling (2.05 → 0.86) while retrieval plateaued from
  ~ep8 — exactly the memorization signature in finding 5 (loss improves,
  top-k stalls). More epochs won't fix it.
- Likely bottleneck is the input, not the encoder: top-80 truncation throws
  away the weak-peak tail, while the binned MLP sees the full spectrum
  (12k L2-normalized bins). The subformula flags did not compensate.
- Recommendation: next attempt should give the set encoder full-peak
  coverage (all peaks, intensity-weighted pooling) or a hybrid
  set + binned-residual input — not more epochs of this config.

## Files

- `v-fpnet/train_fpnet.py` — featurization, model, training, retrieval eval
  (`--epochs N`, `--resume`, `--init CKPT`)
- `v-fpnet/baseline_eval.py` — v2/v8 under the identical protocol
- `v-fpnet/fpnet_best.pt` + `v-fpnet/meta.json` — best-pooled-@5 checkpoint
- `v-fpnet/fpnet_last.pt` — resume state
- `v-fpnet/subfrag_masses.npy` — subformula mass library (28,939)
- `v-fpnet/train.log`, `v-fpnet/train2.log` — phase-1 / phase-2 logs
