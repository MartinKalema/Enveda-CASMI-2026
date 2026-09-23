# v10: cleaned floor (per-adduct index + denoising), v8 fills unchanged

Plain English: every point we ever scored came from the cosine top-5, and the
clean track measured two floor upgrades: compare same-adduct spectra only
(0.445 vs 0.058 cross-adduct) and denoise peaks (0.01 floor + top-200 keeps
truth rank 30->16). v10 changes ONLY the floor; fills stay v8's fingerprint
channel so the LB delta isolates the floor effect.
