"""Evaluate v2/v8 binned-MLP baselines under the SAME fpnet protocol.

Same splits (seed), same pools, same queries, same raw f.z ranking, so the
pooled-truth and disjoint numbers are directly comparable to v-fpnet.
"""
import importlib.util
import numpy as np
import torch

PROJECT = "/Users/martin/Desktop/enveda-casmi26-molecule-id"
import sys
sys.path.insert(0, PROJECT)


def load_fpnet_mod():
    spec = importlib.util.spec_from_file_location(
        "train_fpnet", f"{PROJECT}/v-fpnet/train_fpnet.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(seed=0):
    t = load_fpnet_mod()
    from v2.train_fp import FpMLP
    from v8.train_fpt import featurize as mlp_featurize
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tr, pool, vdis, fmap = t.make_splits(seed)
    pool_m, pool_s, pool_P = t.build_pool(pool, fmap)
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(pool), size=min(150000, len(pool)), replace=False)
    train_idx.sort()
    rest = pool.drop(pool.index[train_idx])
    q_pooled = rest.sample(800, random_state=seed)
    q_dis = vdis.sample(800, random_state=seed)
    fmap_f = {s: np.unpackbits(p).astype(np.float32) for s, p in fmap.items()}
    for name, pt, dh in [("v2", f"{PROJECT}/v2/fp_mlp.pt", 1024),
                         ("v8", f"{PROJECT}/v8/fp_trans.pt", 1536)]:
        model = FpMLP(d_h=dh).to(device)
        model.load_state_dict(torch.load(pt, map_location=device))
        model.eval()
        with torch.no_grad():
            Zp = model(torch.from_numpy(mlp_featurize(q_pooled)).to(device)).cpu().numpy()
            Zd = model(torch.from_numpy(mlp_featurize(q_dis)).to(device)).cpu().numpy()
        hp, np_ = t.hits_at_k(Zp, q_pooled, pool_m, pool_s, pool_P, fmap_f,
                              inject_truth=False)
        hd, nd = t.hits_at_k(Zd, q_dis, pool_m, pool_s, pool_P, fmap_f,
                             inject_truth=True)
        print(f"{name}: pooled@1/5/10={hp[1]:.3f}/{hp[5]:.3f}/{hp[10]:.3f} (n={np_}) "
              f"disjoint@1/5/10={hd[1]:.3f}/{hd[5]:.3f}/{hd[10]:.3f} (n={nd})",
              flush=True)


if __name__ == "__main__":
    main()
