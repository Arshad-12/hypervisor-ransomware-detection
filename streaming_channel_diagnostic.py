"""
streaming_channel_diagnostic.py — Is the Streaming Channel Redundant or Broken?
================================================================================================
Per HANDOFF item 63 (August 17, 2026).

Component ablation (item 58 Task A) showed classifier+streaming is IDENTICAL to classifier
alone (86% variant recall, 10% Conti_06, 0% FPR, both rows) — streaming fired on 0 executions
in every logged run (FULL_LOFO_FINAL, BILSTM_LOFO_RETRAINED, RANSAP_CROSS_DATASET, every
Conti_06 per-exec table). Removal is provably lossless on this data and needs no re-run — if
`s` never reaches STREAM_THRESH=0.65, the fusion OR-rule reduces to just the head's verdict.
What's missing is WHY it never fires, which decides how this gets written up:
  (a) REDUNDANT — binary fusion superseded it (it mattered only in the old 3-way Idle/Benign/
      Ransomware design where a firing head without primary escalation mapped to Idle, not
      Attack).
  (b) BROKEN — `streaming_max_2way` zero-pads each 50-window chunk back out to 300 windows
      before scoring, an input distribution the CNN never saw in training, potentially
      destroying the signal outright.

★ NO RETRAINING. Frozen classifier_2way.pt, forward passes only.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
STREAM_THRESH = 0.65
N_WINDOWS = 300
W = 50

RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES     = ["Firefox", "Office", "Idle"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        return super().default(o)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Load data + frozen classifier
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72); log("Loading data + frozen classifier_2way.pt"); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
text_seqs  = np.load(FEATURES_DIR / "text_sequences.npy")
split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]

val_ben_idx  = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
val_ran_idx  = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])

log(f"  Val Benign={len(val_ben_idx)}  Val LEnc={len(val_lenc_idx)}  "
    f"Val Ransomware={len(val_ran_idx)}  Variants={len(variant_idx)}")

class TextEmbeddingClassifier(nn.Module):
    def __init__(self, emb_dim=384, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.classifier    = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_classes))
        self.reconstructor = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, emb_dim))
    def forward(self, x):
        feat   = self.conv(x.permute(0, 2, 1))
        pooled = feat.mean(dim=-1)
        return self.classifier(pooled), self.reconstructor(pooled), pooled

clf2 = TextEmbeddingClassifier(emb_dim=384, n_classes=2).to(device)
clf2.load_state_dict(torch.load(FEATURES_DIR / "classifier_2way.pt", map_location=device))
clf2.eval()
log("  classifier_2way.pt loaded (frozen, no retraining).")

@torch.no_grad()
def primary_p(i):
    x = torch.tensor(text_seqs[int(i)][None], dtype=torch.float32).to(device)
    logits, _, _ = clf2(x)
    return torch.softmax(logits, dim=1)[0, 1].item()

@torch.no_grad()
def streaming_max_2way(i):
    """Exactly as implemented throughout the project: 50-window chunks, zero-padded back to
    300 windows, max P(ransomware) over all chunks. Computed UNCONDITIONALLY here (not gated
    on primary p < 0.5) so we can probe confidently-ransomware executions too."""
    seq = text_seqs[int(i)]; best = 0.0
    for start in range(0, N_WINDOWS - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((N_WINDOWS - W, 384), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = clf2(x)
        p = torch.softmax(logits, dim=1)[0, 1].item()
        if p > best: best = p
    return best

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Streaming score distribution per class group
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Streaming score distribution — computed unconditionally, all groups"); log("=" * 72)

groups = {
    "Val Benign": val_ben_idx, "Val LEnc": val_lenc_idx,
    "Val Ransomware": val_ran_idx, "Conti variants (70)": variant_idx,
}

group_scores = {}
for name, idx_arr in groups.items():
    log(f"  Computing streaming for {name} (n={len(idx_arr)})...")
    scores = np.array([streaming_max_2way(i) for i in idx_arr])
    group_scores[name] = scores

log(f"\n  {'Group':<22} {'N':>4} {'Min':>8} {'Mean':>8} {'P50':>8} {'P95':>8} {'Max':>8}")
log("  " + "-" * 68)
for name, scores in group_scores.items():
    stats = {"n": len(scores), "min": float(scores.min()), "mean": float(scores.mean()),
              "p50": float(np.percentile(scores, 50)), "p95": float(np.percentile(scores, 95)),
              "max": float(scores.max())}
    log(f"  {name:<22} {stats['n']:>4} {stats['min']:>8.4f} {stats['mean']:>8.4f} "
        f"{stats['p50']:>8.4f} {stats['p95']:>8.4f} {stats['max']:>8.4f}")
    json_out[name] = stats


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — How far below threshold does it operate?
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Counts exceeding candidate thresholds (per group)"); log("=" * 72)
thresholds = [0.65, 0.5, 0.3, 0.1]
log(f"  {'Group':<22} " + "  ".join(f">{t:<5}" for t in thresholds))
log("  " + "-" * 68)
thresh_json = {}
for name, scores in group_scores.items():
    counts = [int((scores > t).sum()) for t in thresholds]
    log(f"  {name:<22} " + "  ".join(f"{c:>5}" for c in counts))
    thresh_json[name] = dict(zip([str(t) for t in thresholds], counts))
json_out["threshold_counts"] = thresh_json

all_scores_combined = np.concatenate(list(group_scores.values()))
log(f"\n  Combined (n={len(all_scores_combined)}): " +
    "  ".join(f">{t}: {int((all_scores_combined > t).sum())}" for t in thresholds))
json_out["combined_threshold_counts"] = {str(t): int((all_scores_combined > t).sum()) for t in thresholds}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Padding probe: confidently-ransomware executions
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("PADDING PROBE — confidently-ransomware executions (p > 0.9)"); log("=" * 72)

probe_pool = np.concatenate([val_ran_idx, variant_idx])
log(f"  Computing primary P(ransomware) for {len(probe_pool)} candidate execs...")
p_all = {int(i): primary_p(i) for i in probe_pool}
confident = sorted([i for i in probe_pool if p_all[int(i)] > 0.9], key=lambda i: -p_all[int(i)])
log(f"  Found {len(confident)} executions with p > 0.9 (using first 10)")
probe_idx = confident[:10]

log(f"\n  {'Idx':>6} {'Label':<12} {'p_ran':>8} {'streaming':>10}")
probe_rows = []
for i in probe_idx:
    i = int(i)
    s = streaming_max_2way(i)
    lbl = all_labels[i]
    log(f"  {i:>6} {lbl:<12} {p_all[i]:>8.4f} {s:>10.4f}")
    probe_rows.append({"idx": i, "label": str(lbl), "p_ran": p_all[i], "streaming": s})
json_out["padding_probe"] = probe_rows

if probe_rows:
    mean_probe_stream = np.mean([r["streaming"] for r in probe_rows])
    log(f"\n  Mean streaming score on these {len(probe_rows)} confidently-ransomware execs: {mean_probe_stream:.4f}")
    if mean_probe_stream < 0.1:
        log("  -> Near-zero even on confident ransomware. Consistent with explanation (b) BROKEN:")
        log("     zero-padding a 50-window chunk to 300 windows produces an input distribution")
        log("     the CNN never saw in training, destroying the signal regardless of content.")
    elif mean_probe_stream >= 0.3:
        log("  -> Meaningful scores (0.3+) on confident ransomware, just below STREAM_THRESH=0.65.")
        log("     Consistent with explanation (a) REDUNDANT at the current threshold — the")
        log("     channel still carries signal, it's just not needed at this operating point.")
    else:
        log("  -> Ambiguous zone (0.1-0.3) — some signal present but weak.")
    json_out["padding_probe_mean"] = float(mean_probe_stream)

log()
log("=" * 72)
log("READ: near-zero scores including on confident ransomware -> BROKEN (write up as an")
log("implementation limitation; the earlier 3-way NB15 result used a different implementation).")
log("Meaningful (0.3-0.5) but below 0.65 -> REDUNDANT at this threshold (could be recovered by")
log("recalibration but adds nothing at the current 5% FPR operating point).")
log("=" * 72)

out_txt  = FEATURES_DIR / "streaming_channel_diagnostic_results.txt"
out_json = FEATURES_DIR / "streaming_channel_diagnostic_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("streaming_channel_diagnostic.py done.")
