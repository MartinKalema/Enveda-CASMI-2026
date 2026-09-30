"""TDD for the fork-FPNet harness: every piece must exist and behave
before any head-to-head number counts."""
import numpy as np


def test_harness_has_all_pieces():
    import v13.fork_fpnet_raw as F
    for name in ("SinEmb", "Block", "FPNet", "prep_peaks", "model_logits",
                 "_fp_init", "fp_and_mass", "ADDUCT_LIST", "instr_family"):
        assert hasattr(F, name), f"missing {name}"


def test_bits_mask_shape():
    import v13.fork_fpnet_raw as F
    bits = np.load("/tmp/cocofp/fp_bits.npy")
    assert bits.shape == (6930,) and bits.max() < 4096 + 4096 + 2048 + 167, \
        f"bit-index array unexpected: {bits.shape}"


def test_fp_and_mass_shape():
    import v13.fork_fpnet_raw as F
    F._fp_init()
    out = F.fp_and_mass("CCO")
    assert out is not None
    fp, mass = out
    assert len(fp) == 6930, f"expected 6930-bit fp, got {len(fp)}"
    assert abs(mass - 46.07) < 0.5


def test_model_logits_finite():
    import torch
    import pandas as pd
    import v13.fork_fpnet_raw as F
    ck = torch.load("/tmp/fpmodels/fp_single_s2.pt", map_location="cpu", weights_only=False)
    net = F.FPNet(ck["nbits"], d=ck["d"], layers=ck["layers"]).eval()
    net.load_state_dict(ck["model"])
    F._MODEL = ([net], [], "cpu", ck["nbits"])
    tr = pd.read_parquet(
        "/Users/martin/Desktop/enveda-casmi26-molecule-id/data/train.parquet",
        columns=["normalized_smiles", "adduct", "precursor_mz", "ms2_mzs",
                 "ms2_normalized_intensities", "instrument_type",
                 "collision_energy_ev", "ionization_mode"]).head(1)
    r = tr.iloc[0]
    z = F._logits_raw([(r["ms2_mzs"], r["ms2_normalized_intensities"])], [net],
                      r["precursor_mz"], r["adduct"], r["instrument_type"],
                      r["collision_energy_ev"], 1.0)
    assert z is not None and np.all(np.isfinite(z)) and len(z) == 6930
