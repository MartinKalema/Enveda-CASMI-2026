"""Formula candidates from structures (database-dependent, tractable).
De novo enumeration is deferred: v1 ranks formulae observed among
mass-window candidate structures by aggregated explained intensity.
"""
import numpy as np
from collections import Counter


class FormulaPrior:
    """FastFilter-lite: P(formula) from train frequencies."""

    def __init__(self):
        self.counts = Counter()
        self.total = 0

    def fit(self, formulae):
        self.counts.update(formulae)
        self.total = sum(self.counts.values())

    def score(self, fstr):
        return float(np.log((self.counts.get(fstr, 0) + 1) / (self.total + len(self.counts) + 1)))
