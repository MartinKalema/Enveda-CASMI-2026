"""Grid search GBM hyperparameters on cached features (3-fold seed holdout).
Compares: mine-made-up (300/15/0.05/10) vs theirs (500/depth6/0.03/leaf80/l2-1)
vs small grid. TDD: asserts best config beats both anchors before shipping.
"""
