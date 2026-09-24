# Head-to-head: every channel, identical frozen queries (3 seeds x 50)

Plain English: all past channel numbers came from different splits and can't
be compared. This runs cosine, entropy, analog, explained-intensity,
MetFrag, and fingerprint-dot on the SAME queries and candidates, one table.
Winner earns production slots; losers are documented, not debated.

## Head-to-head (150 queries, 3 seeds, identical candidates)
cos 0.003, ent 0.008, ana 0.031, exp 0.087, metfrag 0.041, fpdot 0.350.
Fingerprint-dot dominates 4x over the next channel. All fills rank by fpdot;
tautomer dedup + exp/metfrag cascades are the only remaining fill-side levers.
