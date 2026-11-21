"""
cerber_diagnosis.py — Cerber Miss Diagnosis (RanSAP cross-dataset)
===================================================================
Two diagnostics:

(1) ENTROPY TRAJECTORY PLOT
    Plot mean H_sw (Shannon entropy, col 4 of feat5) across 300 windows for:
      - Cerber (the missed family)
      - GandCrab4 + TeslaCrypt (novel families that WERE caught, same storage-only test)
      - RanSMAP training families (Darkside, Ryuk, WannaCry, REvil/Sodinokibi)
      - Benign (Excel, Firefox) and LegitEnc (Zip, SDelete, AESCrypt)
    Saved to /scratch/aabdulj4/logs/cerber_entropy_plot.png

(2) STORAGE-ONLY vs FULL PIPELINE on RanSMAP
    Run 6-fold LOFO on RanSMAP using storage-only 5-dim BiLSTM (no CNN, no streaming)
    to quantify how much the memory modality contributes to cross-family detection.
    Compare to full pipeline LOFO numbers (100% all 6 families from FULL_LOFO_FINAL).
    This answers: "Is Cerber's miss a general storage-only limitation, or specific to Cerber?"

Outputs:
    /scratch/aabdulj4/logs/cerber_entropy_plot.png
    /scratch/aabdulj4/logs/cerber_diagnosis.log
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

RANSAP_FEAT  = Path("/scratch/aabdulj4/RanSAP/features")
RANSMAP_FEAT = Path("/scratch/aabdulj4/RanSMAP/features")
LOG_DIR      = Path("/scratch/aabdulj4/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Device: {device}")

# ── Load RanSAP feat5 ─────────────────────────────────────────────────────────
import re
def norm_label(lbl):
    return re.sub(r'-(largefiles|w10dirs.*)$', '', lbl)

rsp_feat5    = np.load(RANSAP_FEAT / "ransap_feat5.npy")     # (1495, 300, 5)
rsp_labels_raw = np.load(RANSAP_FEAT / "ransap_labels.npy", allow_pickle=True).astype(str)
rsp_labels   = np.array([norm_label(l) for l in rsp_labels_raw])
H_SW_COL     = 4   # H_sw = Shannon entropy of writes

log(f"RanSAP: {rsp_feat5.shape}, labels: {np.unique(rsp_labels).tolist()}")

# ── Load RanSMAP feat5 (storage cols only) ────────────────────────────────────
smap_meta   = pd.read_csv(RANSMAP_FEAT / "metadata.csv")
smap_labels = np.load(RANSMAP_FEAT / "text_labels.npy", allow_pickle=True).astype(str)
split       = np.load(RANSMAP_FEAT / "fewshot_split_cnn.npz", allow_pickle=True)
train_idx   = split["train_idx"]

log(f"RanSMAP: {len(smap_meta)} execs, loading feat_23...")
smap_feat23 = np.zeros((len(smap_meta), 300, 23), dtype=np.float32)
for ii, row in smap_meta.iterrows():
    fp = row.get("filepath", row.get("feat_path", None))
    if fp and Path(fp).exists():
        smap_feat23[ii] = np.load(fp)
smap_feat5 = smap_feat23[:, :, :5]
log(f"RanSMAP feat5: {smap_feat5.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: Entropy trajectory plot
# ═══════════════════════════════════════════════════════════════════════════════
log()
log("=" * 70)
log("DIAGNOSTIC 1: H_sw entropy trajectory comparison")
log("=" * 70)

# Groups to plot
GROUPS = {
    # RanSAP families
    "Cerber (missed, novel)"        : ("ransap", ["Cerber"],    "red",       "--", 3.0),
    "GandCrab4 (caught, novel)"     : ("ransap", ["GandCrab4"], "orange",    "-",  1.5),
    "TeslaCrypt (caught, novel)"    : ("ransap", ["TeslaCrypt"],"darkorange","-",  1.5),
    "Sodinokibi (caught, overlap)"  : ("ransap", ["Sodinokibi"],"purple",    "-",  1.5),
    "WannaCry (caught, overlap)"    : ("ransap", ["WannaCry"],  "blue",      "-",  1.5),
    "Darkside (caught, overlap)"    : ("ransap", ["Darkside"],  "navy",      "-",  1.5),
    "Ryuk (caught, overlap)"        : ("ransap", ["Ryuk"],      "steelblue", "-",  1.5),
    # RanSAP benign (for FPR reference)
    "Zip (FP, legit-enc)"           : ("ransap", ["Zip"],       "green",     ":",  1.5),
    "AESCrypt (FP, legit-enc)"      : ("ransap", ["AESCrypt"],  "limegreen", ":",  1.5),
    "SDelete (low-FP)"              : ("ransap", ["SDelete"],   "olive",     ":",  1.5),
    "Excel/Firefox (low-FP)"        : ("ransap", ["Excel","Firefox"], "gray",":",  1.5),
}

windows = np.arange(300)
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax1 = axes[0]
ax1.set_title("Mean H_sw Trajectory: Cerber vs Caught Novel Families (RanSAP)", fontsize=13)
focus_groups = ["Cerber (missed, novel)", "GandCrab4 (caught, novel)",
                "TeslaCrypt (caught, novel)", "Zip (FP, legit-enc)", "Excel/Firefox (low-FP)"]

ax2 = axes[1]
ax2.set_title("Mean H_sw Trajectory: All RanSAP Families + Benign", fontsize=13)

for label, (src, families, color, ls, lw) in GROUPS.items():
    if src == "ransap":
        mask = np.isin(rsp_labels, families)
        seqs = rsp_feat5[mask, :, H_SW_COL]   # (n, 300)
    else:
        mask = np.isin(smap_labels, families)
        seqs = smap_feat5[mask, :, H_SW_COL]

    if mask.sum() == 0:
        log(f"  WARNING: no execs for {label}")
        continue

    mean_h = seqs.mean(axis=0)
    std_h  = seqs.std(axis=0)
    n      = mask.sum()
    log(f"  {label}: n={n}, mean_H_sw={mean_h.mean():.3f} ± {std_h.mean():.3f}")

    ax2.plot(windows, mean_h, color=color, linestyle=ls, linewidth=lw,
             label=f"{label} (n={n})")
    ax2.fill_between(windows, mean_h - std_h, mean_h + std_h,
                     alpha=0.08, color=color)

    if label in focus_groups:
        ax1.plot(windows, mean_h, color=color, linestyle=ls, linewidth=lw,
                 label=f"{label} (n={n})")
        ax1.fill_between(windows, mean_h - std_h, mean_h + std_h,
                         alpha=0.12, color=color)

for ax in axes:
    ax.set_xlabel("Window index (0.1s each)")
    ax.set_ylabel("Mean Shannon Entropy (H_sw)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 299)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = LOG_DIR / "cerber_entropy_plot.png"
plt.savefig(plot_path, dpi=150)
log(f"\n  Plot saved: {plot_path}")

# ── Numerical summary ─────────────────────────────────────────────────────────
log()
log("  Numerical H_sw summary (mean over 300 windows, per exec):")
log(f"  {'Family':<30} {'N':>5} {'Mean H_sw':>10} {'Std':>8} {'% windows > 0.5':>16}")
log("  " + "-" * 75)
for label, (src, families, *_) in GROUPS.items():
    if src == "ransap":
        mask = np.isin(rsp_labels, families)
        seqs = rsp_feat5[mask, :, H_SW_COL]
    else:
        mask = np.isin(smap_labels, families)
        seqs = smap_feat5[mask, :, H_SW_COL]
    if mask.sum() == 0: continue
    mean_h  = seqs.mean()
    std_h   = seqs.std()
    pct_hi  = (seqs > 0.5).mean() * 100
    log(f"  {label:<30} {mask.sum():>5} {mean_h:>10.3f} {std_h:>8.3f} {pct_hi:>15.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: Storage-only LOFO on RanSMAP (6 families, 3 seeds)
# ═══════════════════════════════════════════════════════════════════════════════
log()
log("=" * 70)
log("DIAGNOSTIC 2: Storage-only 5-dim BiLSTM LOFO on RanSMAP")
log("=" * 70)
log("(Quantifies how much dropping memory degrades detection generally)")

RANSOMWARE_FAMILIES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
NONRAN_FAMILIES     = ["AESCrypt", "SDelete", "Zip", "Firefox", "Office", "Idle"]
SEEDS = [42, 123, 777]
EPOCHS = 50
BATCH  = 32
LR     = 1e-3
FPR_TARGET = 0.05

class BiLSTM5(nn.Module):
    def __init__(self, input_dim=5, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(hidden * 2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        g = self.pool(out.permute(0, 2, 1)).squeeze(-1)
        return torch.sigmoid(self.fc(g)).squeeze(-1)

def is_ran(lbl):
    return any(lbl == r or lbl.startswith(r + "_") or lbl.startswith(r + "-")
               for r in RANSOMWARE_FAMILIES)
def is_ben(lbl):
    return any(lbl == b or lbl.startswith(b + "_") or lbl.startswith(b + "-")
               for b in NONRAN_FAMILIES)

# Use all 1970 execs for LOFO (not just train split)
all_ran_mask = np.array([is_ran(l) for l in smap_labels])
all_ben_mask = np.array([is_ben(l) for l in smap_labels])
log(f"  RanSMAP total ran: {all_ran_mask.sum()}, ben: {all_ben_mask.sum()}")

lofo_results = {}

# Identify 6 core families (not variant groups)
core_families = []
for r in RANSOMWARE_FAMILIES:
    exact_mask = smap_labels == r
    if exact_mask.sum() > 0:
        core_families.append(r)

log(f"  Core families for LOFO: {core_families}")

for held_out in core_families:
    log()
    log(f"  ── Hold out: {held_out} ──")

    # Train: all ran except held-out family (exact + variants) + all ben
    train_ran_mask = np.array([
        is_ran(l) and not (l == held_out or l.startswith(held_out + "_") or l.startswith(held_out + "-"))
        for l in smap_labels
    ])
    train_ben_mask = all_ben_mask.copy()
    test_mask      = np.array([
        l == held_out or l.startswith(held_out + "_") or l.startswith(held_out + "-")
        for l in smap_labels
    ])

    log(f"    Train ran: {train_ran_mask.sum()}, train ben: {train_ben_mask.sum()}, test: {test_mask.sum()}")

    # Scale
    X_tr = np.concatenate([smap_feat5[train_ran_mask], smap_feat5[train_ben_mask]])
    y_tr = np.concatenate([np.ones(train_ran_mask.sum()), np.zeros(train_ben_mask.sum())])
    sc   = StandardScaler()
    X_tr_flat = X_tr.reshape(-1, 5)
    sc.fit(X_tr_flat)
    X_tr_sc = ((X_tr.reshape(-1, 5) - sc.mean_) / (sc.scale_ + 1e-8)).reshape(-1, 300, 5)

    # Train ensemble
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        m = BiLSTM5().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-4)
        Xt = torch.tensor(X_tr_sc, dtype=torch.float32)
        yt = torch.tensor(y_tr,    dtype=torch.float32)
        ds = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True)
        pos_w = train_ben_mask.sum() / max(train_ran_mask.sum(), 1)
        crit  = nn.BCELoss(reduction="none")
        m.train()
        for ep in range(EPOCHS):
            for xb, yb in ds:
                xb, yb = xb.to(device), yb.to(device)
                sc_out = m(xb)
                w = torch.where(yb > 0.5, torch.tensor(pos_w, device=device),
                                torch.tensor(1.0, device=device))
                loss = (crit(sc_out, yb) * w).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        models.append(m)

    # Calibrate threshold on training benign scores
    def score_batch(X_np):
        X_sc = ((X_np.reshape(-1, 5) - sc.mean_) / (sc.scale_ + 1e-8)).reshape(-1, 300, 5)
        all_s = []
        for m in models:
            m.eval()
            with torch.no_grad():
                Xt = torch.tensor(X_sc, dtype=torch.float32).to(device)
                all_s.append(m(Xt).cpu().numpy())
        return np.stack(all_s).mean(axis=0)

    ben_scores = score_batch(smap_feat5[train_ben_mask])
    lstm_thresh = float(np.percentile(ben_scores, 100 * (1 - FPR_TARGET)))

    # Score test (held-out family)
    test_scores = score_batch(smap_feat5[test_mask])
    n_det = (test_scores > lstm_thresh).sum()
    recall = n_det / test_mask.sum()

    # Val FPR
    n_fp  = (ben_scores > lstm_thresh).sum()
    fpr   = n_fp / len(ben_scores)

    lofo_results[held_out] = {"n": test_mask.sum(), "n_det": int(n_det),
                               "recall": recall, "fpr": fpr}
    log(f"    {held_out}: {n_det}/{test_mask.sum()} detected ({100*recall:.1f}%), FPR={100*fpr:.1f}%")

log()
log("  Storage-only LOFO Summary (5-dim BiLSTM only, no CNN, no streaming):")
log(f"  {'Family':<12} {'N':>5} {'Det':>5} {'Recall':>8} {'FPR':>8}")
log("  " + "-" * 45)
all_det = 0; all_n = 0
for fam in core_families:
    r = lofo_results.get(fam, {})
    if not r: continue
    log(f"  {fam:<12} {r['n']:>5} {r['n_det']:>5} {100*r['recall']:>7.1f}% {100*r['fpr']:>7.1f}%")
    all_det += r["n_det"]; all_n += r["n"]
log("  " + "-" * 45)
log(f"  {'MEAN':<12} {all_n:>5} {all_det:>5} {100*all_det/all_n if all_n else 0:>7.1f}%")
log()
log("  FULL PIPELINE LOFO reference (from FULL_LOFO_FINAL, all 6 families): 100% each")
log("  → Gap between storage-only and full pipeline = memory feature contribution")

# ── Save log ──────────────────────────────────────────────────────────────────
(LOG_DIR / "cerber_diagnosis.log").write_text("\n".join(lines))
log()
log("cerber_diagnosis.py done.")
