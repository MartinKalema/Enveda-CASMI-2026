"""FP + BDE artifacts for the ORIGINAL-COCONUT universe.
Reuses per-SMILES values from existing caches; computes only the missing.
Out: data/orig_fp_packed.npy, data/orig_fp_keys.pkl,
     data/orig_bde_{pos,neg,na}.pkl
"""
import pickle

import numpy as np

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
REGIMES = {"pos": "[M+H]+", "neg": "[M-H]-", "na": "[M+Na]+"}


def _fp_one(s):
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    import numpy as np
    _g = _fp_one.g
    try:
        m = Chem.MolFromSmiles(s)
        v = np.concatenate([_g[0].GetFingerprintAsNumPy(m).astype(np.uint8),
                            _g[1].GetFingerprintAsNumPy(m).astype(np.uint8),
                            _g[2].GetFingerprintAsNumPy(m).astype(np.uint8),
                            np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])
        return s, v[_g[3]]
    except Exception:
        return s, None


def _fp_init():
    from rdkit.Chem import rdFingerprintGenerator
    import numpy as np
    _fp_one.g = (
        rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096),
        rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096),
        rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6),
        np.load("/tmp/cocofp/fp_bits.npy"),
    )


def _bde_one(args):
    import numpy as np
    from v13.frag_up import fragment_masses_bde
    s, adduct = args
    try:
        return s, np.asarray(fragment_masses_bde(s, adduct), dtype=float)
    except Exception:
        return s, np.zeros(0)


def main():
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from rdkit.Chem import rdFingerprintGenerator
    uni = pickle.load(open(f"{PROJECT}/data/orig_universe.pkl", "rb"))
    print("universe:", len(uni), flush=True)
    # ---- FP: reuse old rows ----
    old_keys = pickle.load(open(f"{PROJECT}/data/their_fp_keys.pkl", "rb"))
    old_M = np.load(f"{PROJECT}/data/their_fp_packed.npy", mmap_mode="r")
    old = {s: np.array(old_M[i]) for i, s in enumerate(old_keys)}
    del old_M
    missing = [s for s in uni if s not in old]
    print(f"fp reuse {len(uni) - len(missing)}/{len(uni)}", flush=True)
    if missing:
        with ProcessPoolExecutor(max_workers=8, initializer=_fp_init) as ex:
            for i, (s, v) in enumerate(ex.map(_fp_one, missing, chunksize=50)):
                if v is not None:
                    old[s] = np.packbits(v[:6930])
                if (i + 1) % 20000 == 0:
                    print(f"fp {i + 1}/{len(missing)}", flush=True)
    M = np.zeros((len(uni), 867), dtype=np.uint8)
    dropped = 0
    for i, s in enumerate(uni):
        v = old.get(s)
        if v is None:
            dropped += 1
            continue
        M[i] = v
    print("fp unparseable:", dropped, flush=True)
    np.save(f"{PROJECT}/data/orig_fp_packed.npy", M)
    pickle.dump(uni, open(f"{PROJECT}/data/orig_fp_keys.pkl", "wb"))
    print("saved fp", M.shape, flush=True)
    # ---- BDE: reuse old dicts ----
    prev = {}
    for _reg in REGIMES:
        with open(f"{PROJECT}/data/bde_frag_{_reg}.pkl", "rb") as f:
            prev[_reg] = pickle.load(f)
    for _reg, adduct in REGIMES.items():
        out = {}
        miss = [s for s in uni if s not in prev[_reg]]
        for s in uni:
            if s in prev[_reg]:
                out[s] = prev[_reg][s]
        print(f"bde {_reg} reuse {len(out)}/{len(uni)}", flush=True)
        if miss:
            jobs = [(s, adduct) for s in miss]
            with ProcessPoolExecutor(max_workers=8) as ex:
                for i, (s, f) in enumerate(ex.map(_bde_one, jobs, chunksize=50)):
                    out[s] = f
                    if (i + 1) % 20000 == 0:
                        print(f"bde {_reg} {i + 1}/{len(miss)}", flush=True)
        with open(f"{PROJECT}/data/orig_bde_{_reg}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"saved bde {_reg}: {len(out)}", flush=True)


if __name__ == "__main__":
    main()
