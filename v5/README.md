# v5: calibrated multi-seed validation + hybrid channels (in progress)

Plain English: v4 taught us single-seed validation lies. v5 scores every
channel (cosine, entropy, analog, explained) on the SAME queries across
3 seeds x 50 queries, always with v0-cosine as the anchor. A channel only
joins production if it beats the anchor on a split where the anchor itself
is stable. Feature cache on disk so mixes are evaluated without recompute.
