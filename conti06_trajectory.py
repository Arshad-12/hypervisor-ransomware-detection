"""
Conti_06 Trajectory Check
==========================
Plots write-entropy (H_sw, feature index 4) across all 300 windows
for all 10 Conti_06 executions (8 missed + 2 detected).

Key question: do the 8 missed executions ever spike to high entropy
within the 30-second capture window? If not → no model can detect them
(the ransomware behavior simply hasn't started yet). If yes → detectable
but the current model misses the onset.

Also plots for reference:
  - 3 canonical Conti training executions (high entropy baseline)
  - 3 benign Firefox executions (low entropy baseline)

Runs on login node (CPU only). Saves: conti06_trajectory.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
H_SW = 4    # Shannon entropy of written sectors (the key signal)
H_MW = 5    # memory write entropy (secondary)

meta = pd.read_csv(FEATURES_DIR / "metadata.csv")

def load_exec(idx):
    fp = meta.loc[idx, "filepath"]
    return np.load(fp)   # (300, 23)

# ── Identify Conti_06 and reference executions ────────────────────────────────
c06_idx   = meta.index[meta["label"] == "Conti_06"].tolist()
conti_idx = meta.index[meta["label"] == "Conti"].tolist()[:3]
benign_idx= meta.index[meta["label"] == "Firefox"].tolist()[:3]

# Known detected vs missed (from fusion_diagnose output)
DETECTED = {1491, 1497}   # primary=Attack
MISSED   = {1490, 1492, 1493, 1494, 1495, 1496, 1498, 1499}

print(f"Conti_06 executions: {c06_idx}")
print(f"Detected: {sorted(DETECTED & set(c06_idx))}")
print(f"Missed:   {sorted(MISSED   & set(c06_idx))}")

# ── Load features ─────────────────────────────────────────────────────────────
c06_data = {i: load_exec(i) for i in c06_idx}
ref_conti  = [load_exec(i) for i in conti_idx]
ref_benign = [load_exec(i) for i in benign_idx]

windows = np.arange(300) * 0.1   # time axis in seconds

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
fig.suptitle("Conti_06 Write-Entropy Trajectory\n(H_sw across 300 × 0.1s windows = 30s capture)",
             fontsize=13, fontweight="bold")

# Row 0: 2 detected Conti_06 + 1 canonical Conti
detected_list = sorted(DETECTED & set(c06_idx))
for col, idx in enumerate(detected_list[:2]):
    ax = axes[0, col]
    ax.plot(windows, c06_data[idx][:, H_SW], color="crimson", lw=1.2)
    ax.set_title(f"Conti_06 exec {idx} — DETECTED ✓", color="crimson", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("H_sw entropy")

ax = axes[0, 2]
ax.plot(windows, ref_conti[0][:, H_SW], color="darkred", lw=1.2)
ax.set_title("Canonical Conti (training) — ref", color="darkred", fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("Time (s)"); ax.set_ylabel("H_sw entropy")

# Rows 1–3: 8 missed Conti_06 executions
missed_list = sorted(MISSED & set(c06_idx))
for i, idx in enumerate(missed_list):
    row, col = 1 + i // 3, i % 3
    ax = axes[row, col]
    seq = c06_data[idx][:, H_SW]
    max_h = seq.max()
    color = "darkorange" if max_h > 0.5 else "steelblue"
    ax.plot(windows, seq, color=color, lw=1.2)
    ax.set_title(f"Conti_06 exec {idx} — MISSED ✗\n(max H_sw={max_h:.3f})",
                 color=color, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5, label="0.5 threshold")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("H_sw entropy")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = FEATURES_DIR / "conti06_trajectory_full.png"
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

# ── Text summary ──────────────────────────────────────────────────────────────
print("\n── Trajectory summary ──")
print(f"{'Exec':>6}  {'Status':>10}  {'Max H_sw':>10}  {'Mean H_sw':>11}  {'Spikes>0.5?':>12}  {'When?'}")
print("-" * 75)
for idx in sorted(c06_idx):
    seq = c06_data[idx][:, H_SW]
    status = "DETECTED" if idx in DETECTED else "missed"
    spikes = np.where(seq > 0.5)[0]
    spike_str = f"window {spikes[0]} ({spikes[0]*0.1:.1f}s)" if len(spikes) > 0 else "never"
    print(f"{idx:>6}  {status:>10}  {seq.max():>10.4f}  {seq.mean():>11.4f}  "
          f"{'YES' if len(spikes)>0 else 'NO':>12}  {spike_str}")

dormant = [i for i in missed_list if c06_data[i][:, H_SW].max() <= 0.5]
spikers  = [i for i in missed_list if c06_data[i][:, H_SW].max() >  0.5]
print(f"\nMissed + dormant (never spike above 0.5): {dormant}  n={len(dormant)}")
print(f"Missed + spike (detectable in window):    {spikers}  n={len(spikers)}")

if dormant:
    print(f"\n→ {len(dormant)} executions genuinely dormant within 30s — document as limitation")
if spikers:
    print(f"→ {len(spikers)} executions spike but are missed — temporal VAE warranted")
