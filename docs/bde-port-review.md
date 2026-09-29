# BDE port: the three pieces side by side

## 1. The codebase that scored 0.328
`kernels/fork-034/notebook.ipynb` — a private fork of the public 0.335
4-channel engine (haideptry V17). Verified 0.328 on the board (v6 restore).
Four channels feed a 31-feature seed-bagged GBM: library search, analog
propagation, MetFrag-style fragmentation, fingerprint transformer.

## 2. The part being replaced: ORIGINAL `fragment_masses` (their frag channel)

```python
def fragment_masses(smi, max_breaks=2, max_bonds=34):
    g = mol_graph(smi)
    if g is None: return np.zeros(0)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds: return np.array([w.sum()])
    out = {w.sum()}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    return np.array(sorted(out))
```

What it does: breaks EVERY single link (then every pair), lists all pieces.
Two flaws: no idea which links break easily (all equal), and molecules with
more than 34 links get ZERO pieces (returns just the whole molecule).

It is called by `frag_scores`, which matches each candidate's piece-list
against the mystery peaks (`explain_score`, ±2 H shifts) and takes the best
match over the molecule's spectra. That score becomes 4 features of the
ranker's 31, and the ranker orders the final top 25.

## 3. The replacement: OUR BDE version (now in the notebook, running as v8)

```python
def fragment_masses(smi, max_breaks=2, max_bonds=34, _adduct="[M+H]+"):
    # ... charge-site atoms from adduct regime ...
    # ... every link scored: bond strength + 8x distance from charge ...
    _scored.sort()
    _keep = [i for _, i in _scored[:14]]   # 14 weakest links only
    # ... break only kept links, singly and in pairs ...
```

What changed, nothing else: weakest 14 links instead of all links; breaks
start near where the charge sits (from the charged form); big molecules get
14 useful breaks instead of zero. Same signature, same caller, same ranker —
the single variable. Head-to-head on identical queries: ours 0.199 / 0.214 /
0.193 vs theirs 0.099 / 0.137 / 0.114.
