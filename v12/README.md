# v12: cleaned floor + formula-funnel/frag fills (pending)

Plain English: v10's floor (top-5 untouched) with fills from the validated
alt pipeline instead of fingerprint dots: group candidates by formula, keep
top-2 formulae by subformula explained-intensity, rank isomers by forward
fragmentation fit. No RDKit at inference (precomputed frag cache).
