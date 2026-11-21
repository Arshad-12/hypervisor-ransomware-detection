"""
check_feat5_scale.py — Quick diagnostic: are RanSMAP feat5 and RanSAP feat5 on the same raw scale?
================================================================================================
ransap_head_comparison.py (HANDOFF item 61) came back with TS-TCC and VAE both firing on
100.0% of EVERY RanSAP category, including benign Firefox/Excel — that's not a real finding,
it smells like a raw-value scale mismatch between the two datasets' storage features (T_sr,
T_sw, V_sr, V_sw are raw counts/volumes, not bounded like H_sw is). BiLSTM (sigmoid classifier)
didn't show this failure mode, which fits: a supervised classifier can partially absorb an input
scale mismatch, but an unbounded distance/reconstruction score cannot.

This script just prints per-column raw (UNSCALED) min/mean/max/std for both datasets side by
side. No GPU needed, no training, runs in seconds on the login node or a tiny CPU job.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANSAP_FEAT  = Path("/scratch/aabdulj4/RanSAP/features")
RANSMAP_FEAT = Path("/scratch/aabdulj4/RanSMAP/features")
COLS = ["T_sr", "T_sw", "V_sr", "V_sw", "H_sw"]

print("Loading RanSMAP feat_23 (cols 0:5)...", flush=True)
smap_meta = pd.read_csv(RANSMAP_FEAT / "metadata.csv")
smap_feat23 = np.zeros((len(smap_meta), 300, 23), dtype=np.float32)
for i, row in smap_meta.iterrows():
    fp = row.get("filepath", row.get("feat_path", None))
    if fp and Path(fp).exists():
        smap_feat23[i] = np.load(fp)
smap_feat5 = smap_feat23[:, :, :5]
del smap_feat23

print("Loading RanSAP feat5...", flush=True)
rsp_feat5 = np.load(RANSAP_FEAT / "ransap_feat5.npy")

print()
print(f"{'Col':<6} {'Dataset':<10} {'Min':>12} {'Mean':>12} {'Max':>12} {'Std':>12}")
print("-" * 68)
for c, name in enumerate(COLS):
    for label, arr in [("RanSMAP", smap_feat5), ("RanSAP", rsp_feat5)]:
        col = arr[:, :, c].astype(np.float64)
        print(f"{name:<6} {label:<10} {col.min():>12.4f} {col.mean():>12.4f} "
              f"{col.max():>12.4f} {col.std():>12.4f}")
    print()

print("READ: if RanSAP's Min/Mean/Max for T_sr/T_sw/V_sr/V_sw are orders of magnitude")
print("different from RanSMAP's (H_sw should already roughly match since it's a bounded")
print("Shannon entropy, 0-1, already cross-compared successfully in CERBER_DIAGNOSIS),")
print("that confirms a raw-scale mismatch — RanSMAP's StandardScaler, fit on RanSMAP only,")
print("would map RanSAP's storage features to extreme outliers, which is exactly what would")
print("make an unsupervised distance/reconstruction head saturate to 100% on every category.")
