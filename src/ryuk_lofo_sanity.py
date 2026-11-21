"""
ryuk_lofo_sanity.py — Ryuk Cross-Family LOFO Sanity Check
==========================================================
Per HANDOFF update (41), task 1.

Historically Ryuk was the one genuine cross-family miss (10% in NB11).
The binary_fusion proper LOFO reported 100% Recall / 5% FPR for Ryuk.
This script confirms that 100% is REAL detection (not anomaly escalation)
by breaking down each Ryuk held-out exec by signal driver:
  - Classifier-driven: 2-way CNN primary P(ran) >= 0.5 → Attack
  - Streaming-driven: primary < 0.5 but streaming score >= 0.65 → Attack
  - BiLSTM-driven:    primary < 0.5, stream < 0.65, BiLSTM fires → Attack
  - Missed:           all signals below threshold → Benign

Steps:
  1. Re-run the Ryuk LOFO fold (train 2-way CNN WITHOUT Ryuk, 3 seeds)
  2. Get per-exec verdicts for ALL 150 Ryuk execs (30 val + 120 train)
  3. Break down by driver
  4. Confirm val FPR holds (val non-ran execs at final classifier_2way threshold)
  5. Compare to paired FPR = 7.8% Ben / 2.2% LEnc from binary_fusion LOFO
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
STREAM_THRESH = 0.65
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES     = ["Firefox", "Office", "Idle"]

print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("Loading data")
print("=" * 70)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
split = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
train_idx = split["train_idx"]
val_idx   = split["val_idx"]

val_ben_idx  = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
val_nonran_idx = np.concatenate([val_ben_idx, val_lenc_idx])

# ALL Ryuk execs (train + val) — these are the LOFO hold-out set
all_ryuk_idx = np.array([i for i in range(len(all_labels)) if all_labels[i] == "Ryuk"])
print(f"  Total Ryuk execs: {len(all_ryuk_idx)} (train={sum(i in train_idx for i in all_ryuk_idx)}, val={sum(i in val_idx for i in all_ryuk_idx)})")
print(f"  Val non-ran: {len(val_nonran_idx)} (ben={len(val_ben_idx)}, lenc={len(val_lenc_idx)})", flush=True)

# Load MiniLM sequences
print("  Loading MiniLM sequences...")
minilm_seqs = np.load(FEATURES_DIR / "text_sequences.npy")  # (1970, 300, 384)

# Load BiLSTM scores (from binary_fusion.py, trained on all 1800)
lstm_scores_all = np.load(FEATURES_DIR / "lstm_scores_all.npy")
lstm_thresh     = float(np.load(FEATURES_DIR / "lstm_threshold.npy")[0])
print(f"  BiLSTM threshold: {lstm_thresh:.4f}", flush=True)

# ── 2-way CNN definition ───────────────────────────────────────────────────────
class TextEmbeddingClassifier2Way(nn.Module):
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

# ── Build train set WITHOUT Ryuk ───────────────────────────────────────────────
print()
print("=" * 70)
print("Training 2-way CNN — Ryuk held out")
print("=" * 70)

train_noRyuk = np.array([i for i in train_idx if all_labels[i] != "Ryuk"])
ran_noRyuk   = np.array([i for i in train_noRyuk if all_labels[i] in RANSOMWARE_CLASSES])
nonran_noRyuk= np.array([i for i in train_noRyuk if all_labels[i] not in RANSOMWARE_CLASSES])
print(f"  Train (no Ryuk): {len(train_noRyuk)} (ran={len(ran_noRyuk)}, non-ran={len(nonran_noRyuk)})", flush=True)

# Build labels: Ransomware=1, Non-ran=0
def build_2way_dataset(idx_arr):
    X = torch.tensor(minilm_seqs[idx_arr], dtype=torch.float32)
    y = torch.tensor(
        [1 if all_labels[int(i)] in RANSOMWARE_CLASSES else 0 for i in idx_arr],
        dtype=torch.long)
    return X, y

X_tr, y_tr = build_2way_dataset(train_noRyuk)
X_val, y_val = build_2way_dataset(val_idx)

# Balanced sampler
weights = torch.where(y_tr == 1,
    torch.tensor(1.0 / len(ran_noRyuk)),
    torch.tensor(1.0 / len(nonran_noRyuk)))
sampler = WeightedRandomSampler(weights, num_samples=len(y_tr), replacement=True)
tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, sampler=sampler)
va_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

SEEDS = [42, 123, 7]
EPOCHS = 80
best_acc = -1; best_state = None

for seed in SEEDS:
    torch.manual_seed(seed)
    clf = TextEmbeddingClassifier2Way().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    for ep in range(1, EPOCHS + 1):
        clf.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits, _, _ = clf(Xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 20 == 0: print(f"  Seed {seed}: ep {ep}/{EPOCHS}", flush=True)

    clf.eval()
    correct = 0
    with torch.no_grad():
        for Xb, yb in va_loader:
            logits, _, _ = clf(Xb.to(device))
            correct += (logits.argmax(1).cpu() == yb).sum().item()
    acc = correct / len(y_val)
    print(f"  Seed {seed}: val acc = {acc:.1%}", flush=True)
    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.clone() for k, v in clf.state_dict().items()}

print(f"  Best val acc: {best_acc:.1%}")
clf.load_state_dict(best_state)
clf.eval()

# ── Streaming function ─────────────────────────────────────────────────────────
@torch.no_grad()
def streaming_score(idx):
    seq = minilm_seqs[int(idx)]; W = 50; best = 0.0
    for start in range(0, 300 - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = clf(x)
        p = torch.softmax(logits, dim=1)[0, 1].item()  # P(Ransomware)
        if p > best: best = p
    return best

@torch.no_grad()
def primary_p_ran(idx):
    x = torch.tensor(minilm_seqs[int(idx)][None], dtype=torch.float32).to(device)
    logits, _, _ = clf(x)
    return torch.softmax(logits, dim=1)[0, 1].item()

# ── Verdict function (binary, no Idle) ────────────────────────────────────────
def binary_verdict(i, p_ran, s, lstm_score):
    if p_ran >= 0.5: return "Attack", "classifier"
    if s >= STREAM_THRESH: return "Attack", "streaming"
    if lstm_score > lstm_thresh: return "Attack", "bilstm"
    return "Benign", "none"

# ── Evaluate Ryuk held-out execs ───────────────────────────────────────────────
print()
print("=" * 70)
print("Ryuk LOFO held-out — per-execution breakdown")
print("=" * 70)

print("  Computing streaming scores for Ryuk + val non-ran execs...")
rows = []
for idx in all_ryuk_idx:
    i = int(idx)
    p = primary_p_ran(i)
    s = streaming_score(i) if p < 0.5 else 0.0
    lstm_s = lstm_scores_all[i]
    verdict, driver = binary_verdict(i, p, s, lstm_s)
    rows.append({
        "idx": i, "split": "val" if i in val_idx else "train",
        "p_ran": p, "stream": s, "lstm": lstm_s,
        "verdict": verdict, "driver": driver,
    })

print()
print(f"  {'Idx':>6}  {'Split':>5}  {'p_ran':>7}  {'stream':>7}  {'lstm':>7}  {'Verdict':>8}  {'Driver':>12}")
print("  " + "─" * 62)
for r in rows:
    print(f"  {r['idx']:>6}  {r['split']:>5}  {r['p_ran']:>7.4f}  {r['stream']:>7.4f}  "
          f"{r['lstm']:>7.4f}  {r['verdict']:>8}  {r['driver']:>12}")

print()
driver_counts = {}
for r in rows:
    driver_counts[r['driver']] = driver_counts.get(r['driver'], 0) + 1

total   = len(rows)
attacks = sum(1 for r in rows if r['verdict'] == "Attack")
misses  = sum(1 for r in rows if r['verdict'] == "Benign")

print(f"  SUMMARY (N={total} Ryuk execs, all splits):")
print(f"    Attack: {attacks}/{total} = {attacks/total:.0%}")
print(f"    Benign: {misses}/{total} = {misses/total:.0%}")
print(f"    Driver breakdown:")
for driver, cnt in sorted(driver_counts.items()):
    print(f"      {driver:>12}: {cnt}")

# ── Confirm val FPR ────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Val FPR confirmation (on Ryuk-held-out model)")
print("=" * 70)

b_fp = le_fp = 0
for i in val_ben_idx:
    i = int(i)
    p = primary_p_ran(i)
    s = streaming_score(i) if p < 0.5 else 0.0
    v, _ = binary_verdict(i, p, s, lstm_scores_all[i])
    if v == "Attack": b_fp += 1

for i in val_lenc_idx:
    i = int(i)
    p = primary_p_ran(i)
    s = streaming_score(i) if p < 0.5 else 0.0
    v, _ = binary_verdict(i, p, s, lstm_scores_all[i])
    if v == "Attack": le_fp += 1

nr_fp = b_fp + le_fp
print(f"  Val Benign  ({len(val_ben_idx)}): Attack={b_fp} ({b_fp/len(val_ben_idx):.1%})")
print(f"  Val LEnc    ({len(val_lenc_idx)}): Attack={le_fp} ({le_fp/len(val_lenc_idx):.1%})")
print(f"  Combined NR ({len(val_nonran_idx)}): Attack={nr_fp} ({nr_fp/len(val_nonran_idx):.1%})")
print(f"  Expected from binary_fusion LOFO: Ben=7.8%, LEnc=2.2%, NR=5.0%")

print()
print("ryuk_lofo_sanity.py done.")
