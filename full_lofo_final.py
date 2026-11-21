"""
full_lofo_final.py — Full 6-Family LOFO on Final Locked Pipeline
=================================================================
Per HANDOFF update (42), Opus decision July 11 2026.

The cross-family LOFO on record (binary_fusion.py) confirmed 100%/5% FPR
but did NOT include per-exec driver breakdowns. The Ryuk sanity (42) showed
BiLSTM drives 82.7% of Ryuk detections. This script runs the complete 6-fold
LOFO and records the driver breakdown for every family.

Pipeline (LOCKED — per update 40):
  MiniLM 2-way 1D-CNN (retrained per fold, Ransomware vs Non-ran)
  + MiniLM streaming (zero-padded, shared — not retrained)
  + BiLSTM anomaly head (FIXED — lstm_scores_all.npy from binary_fusion)
  → binary fusion: Attack if classifier OR streaming OR BiLSTM fires; else Benign

Per fold:
  1. Hold out one family entirely from 2-way CNN training
  2. Retrain 2-way CNN (3 seeds, 80 epochs)
  3. Evaluate all execs of that family blind (train + val splits)
  4. Record verdict + driver (classifier / streaming / bilstm / none)
  5. Compute val FPR (same val non-ran set, same BiLSTM fixed)

Output: full_lofo_results.txt / .json
"""

import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

FEATURES_DIR  = Path("/scratch/aabdulj4/RanSMAP/features")
STREAM_THRESH = 0.65
SEEDS         = [42, 123, 7]
EPOCHS        = 80

RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES     = ["Firefox", "Office", "Idle"]
LOFO_FAMILIES      = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        return super().default(o)

log(f"Device: {device}")
if device.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load data ──────────────────────────────────────────────────────────────────
log(); log("=" * 72); log("Loading data"); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
split      = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
train_idx  = split["train_idx"]
val_idx    = split["val_idx"]

val_ben_idx  = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
val_nonran_idx = np.concatenate([val_ben_idx, val_lenc_idx])

log(f"  Val non-ran: {len(val_nonran_idx)} (ben={len(val_ben_idx)}, lenc={len(val_lenc_idx)})")
log(f"  Loading MiniLM sequences (1970, 300, 384)...")
minilm_seqs = np.load(FEATURES_DIR / "text_sequences.npy")

log(f"  Loading BiLSTM scores (fixed across all folds)...")
lstm_scores_all = np.load(FEATURES_DIR / "lstm_scores_all.npy")
lstm_thresh     = float(np.load(FEATURES_DIR / "lstm_threshold.npy")[0])
log(f"  BiLSTM threshold: {lstm_thresh:.4f}")

# ── Model ──────────────────────────────────────────────────────────────────────
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

# ── Helpers ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def primary_p_ran(clf, idx):
    x = torch.tensor(minilm_seqs[int(idx)][None], dtype=torch.float32).to(device)
    logits, _, _ = clf(x)
    return torch.softmax(logits, dim=1)[0, 1].item()

@torch.no_grad()
def streaming_score(clf, idx):
    seq = minilm_seqs[int(idx)]; W = 50; best = 0.0
    for start in range(0, 300 - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = clf(x)
        p = torch.softmax(logits, dim=1)[0, 1].item()
        if p > best: best = p
    return best

def binary_verdict(p_ran, s, lstm_score):
    if p_ran >= 0.5:            return "Attack", "classifier"
    if s >= STREAM_THRESH:      return "Attack", "streaming"
    if lstm_score > lstm_thresh:return "Attack", "bilstm"
    return "Benign", "none"

def train_2way_cnn(train_idx_fold):
    """Train 2-way CNN (3 seeds) on given index subset. Returns best model."""
    ran_idx    = [i for i in train_idx_fold if all_labels[i] in RANSOMWARE_CLASSES]
    nonran_idx = [i for i in train_idx_fold if all_labels[i] not in RANSOMWARE_CLASSES]

    X_tr = torch.tensor(minilm_seqs[train_idx_fold], dtype=torch.float32)
    y_tr = torch.tensor(
        [1 if all_labels[int(i)] in RANSOMWARE_CLASSES else 0 for i in train_idx_fold],
        dtype=torch.long)
    X_val = torch.tensor(minilm_seqs[val_idx], dtype=torch.float32)
    y_val = torch.tensor(
        [1 if all_labels[int(i)] in RANSOMWARE_CLASSES else 0 for i in val_idx],
        dtype=torch.long)

    weights = torch.where(y_tr == 1,
        torch.tensor(1.0 / max(len(ran_idx), 1)),
        torch.tensor(1.0 / max(len(nonran_idx), 1)))
    sampler   = WeightedRandomSampler(weights, num_samples=len(y_tr), replacement=True)
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, sampler=sampler)
    va_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

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
            if ep % 20 == 0: log(f"      seed={seed} ep {ep}/{EPOCHS}")
        clf.eval()
        correct = sum((clf(Xb.to(device))[0].argmax(1).cpu() == yb).sum().item()
                      for Xb, yb in va_loader)
        acc = correct / len(y_val)
        log(f"      seed={seed}: val acc = {acc:.1%}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}

    clf.load_state_dict(best_state)
    clf.eval()
    return clf, best_acc

# ── Per-family LOFO ────────────────────────────────────────────────────────────
json_out  = {}
summary   = []

for family in LOFO_FAMILIES:
    log(); log("=" * 72)
    log(f"LOFO: hold out {family}")
    log("=" * 72)

    # All execs of this family (train + val)
    family_idx = np.array([i for i in range(len(all_labels)) if all_labels[i] == family])
    n_train_f  = sum(i in train_idx for i in family_idx)
    n_val_f    = sum(i in val_idx   for i in family_idx)
    log(f"  {family} execs: {len(family_idx)} (train={n_train_f}, val={n_val_f})")

    # Training pool without this family
    train_fold = np.array([i for i in train_idx if all_labels[i] != family])
    ran_fold   = [i for i in train_fold if all_labels[i] in RANSOMWARE_CLASSES]
    log(f"  Training without {family}: {len(train_fold)} execs ({len(ran_fold)} ran)")

    # Retrain 2-way CNN
    clf, best_acc = train_2way_cnn(train_fold)
    log(f"  Best val acc (held-out model): {best_acc:.1%}")

    # Evaluate held-out family
    log(f"  Scoring {len(family_idx)} {family} execs + val FPR...")
    rows = []
    for idx in family_idx:
        i = int(idx)
        p = primary_p_ran(clf, i)
        s = streaming_score(clf, i) if p < 0.5 else 0.0
        lstm_s = lstm_scores_all[i]
        verdict, driver = binary_verdict(p, s, lstm_s)
        rows.append({"idx": i, "p_ran": p, "stream": s, "lstm": lstm_s,
                     "verdict": verdict, "driver": driver})

    n_attack  = sum(1 for r in rows if r["verdict"] == "Attack")
    n_benign  = sum(1 for r in rows if r["verdict"] == "Benign")
    driver_ct = {}
    for r in rows:
        driver_ct[r["driver"]] = driver_ct.get(r["driver"], 0) + 1

    log(f"  {family}: Attack={n_attack}/{len(rows)} ({n_attack/len(rows):.0%})  "
        f"Benign={n_benign}/{len(rows)}")
    log(f"  Driver: " + "  ".join(f"{d}={c}" for d, c in sorted(driver_ct.items())))

    # Val FPR (same val non-ran set, same BiLSTM fixed)
    b_fp = le_fp = 0
    for i in val_ben_idx:
        i = int(i)
        p = primary_p_ran(clf, i)
        s = streaming_score(clf, i) if p < 0.5 else 0.0
        v, _ = binary_verdict(p, s, lstm_scores_all[i])
        if v == "Attack": b_fp += 1
    for i in val_lenc_idx:
        i = int(i)
        p = primary_p_ran(clf, i)
        s = streaming_score(clf, i) if p < 0.5 else 0.0
        v, _ = binary_verdict(p, s, lstm_scores_all[i])
        if v == "Attack": le_fp += 1

    nr_fp = b_fp + le_fp
    log(f"  Val FPR: Ben={b_fp}/{len(val_ben_idx)} ({b_fp/len(val_ben_idx):.1%})  "
        f"LEnc={le_fp}/{len(val_lenc_idx)} ({le_fp/len(val_lenc_idx):.1%})  "
        f"NR={nr_fp}/{len(val_nonran_idx)} ({nr_fp/len(val_nonran_idx):.1%})")

    summary.append({
        "family": family, "N": len(rows),
        "recall": n_attack / len(rows),
        "fpr_ben": b_fp / len(val_ben_idx),
        "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": nr_fp / len(val_nonran_idx),
        "drivers": driver_ct,
    })
    json_out[family] = summary[-1]

    # Print per-exec detail for missed execs (if any)
    missed = [r for r in rows if r["verdict"] == "Benign"]
    if missed:
        log(f"  MISSED execs ({len(missed)}):")
        log(f"    {'Idx':>6}  {'p_ran':>7}  {'stream':>7}  {'lstm':>7}")
        for r in missed:
            log(f"    {r['idx']:>6}  {r['p_ran']:>7.4f}  {r['stream']:>7.4f}  {r['lstm']:>7.4f}")
    else:
        log(f"  No missed execs — 100% recall.")

# ── Summary table ──────────────────────────────────────────────────────────────
log(); log("=" * 72); log("FULL LOFO SUMMARY — FINAL LOCKED PIPELINE"); log("=" * 72)
log(f"  Pipeline: 2-way MiniLM CNN + streaming + BiLSTM (binary, thresh={lstm_thresh:.4f})")
log()
log(f"  {'Family':>12}  {'N':>4}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  "
    f"{'FPR_NR':>8}  {'Classif':>8}  {'Stream':>8}  {'BiLSTM':>8}")
log("  " + "─" * 85)
for s in summary:
    dc = s["drivers"]
    log(f"  {s['family']:>12}  {s['N']:>4}  {s['recall']:>8.0%}  "
        f"{s['fpr_ben']:>9.1%}  {s['fpr_lenc']:>10.1%}  {s['fpr_nr']:>8.1%}  "
        f"{dc.get('classifier',0):>8}  {dc.get('streaming',0):>8}  {dc.get('bilstm',0):>8}")

mean_recall = np.mean([s["recall"] for s in summary])
mean_fpr    = np.mean([s["fpr_nr"] for s in summary])
log("  " + "─" * 85)
log(f"  {'MEAN':>12}  {'':>4}  {mean_recall:>8.0%}  {'':>9}  {'':>10}  {mean_fpr:>8.1%}")

# ── Save ──────────────────────────────────────────────────────────────────────
(FEATURES_DIR / "full_lofo_results.txt").write_text("\n".join(lines))
with open(FEATURES_DIR / "full_lofo_results.json", "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(); log("Saved: full_lofo_results.txt / .json")
log("full_lofo_final.py done.")
