"""
ransap_eval.py — RanSAP Cross-Dataset Validation
=================================================
Two experiments:

  (1) CROSS-DATASET TRANSFER (headline):
      Train storage-only pipeline on RanSMAP → test blind on RanSAP.
      - 5-dim BiLSTM trained on RanSMAP feat_23[:,0:5] (storage features only)
      - MiniLM text CNN = existing classifier_2way.pt (loaded, frozen)
      - Binary fusion: classifier → streaming → bilstm → Benign
      Reports per-family recall + overall FPR on RanSAP.

  (2) INDEPENDENT REPLICATION:
      Leave-one-family-out within RanSAP.
      - Storage-only 5-dim BiLSTM retrained each fold
      - MiniLM text CNN retrained each fold (storage-only text input)
      Reports per-family recall + within-RanSAP FPR.

Assumptions:
  - /scratch/aabdulj4/RanSAP/features/ransap_feat5.npy  exists (from ransap_extract.py)
  - /scratch/aabdulj4/RanSAP/features/ransap_text_seqs.npy exists
  - /scratch/aabdulj4/RanSAP/features/ransap_labels.npy exists
  - /scratch/aabdulj4/ransmap_features/ has RanSMAP data (feat_23 + text_seqs + labels)
  - Trained classifier_2way.pt at ~/ransmap/classifier_2way.pt
  - Trained BiLSTM at ~/ransmap/bilstm_anomaly.pt

Architecture:
  RAN_LABELS    = families treated as ransomware in RanSAP
  BENIGN_LABELS = families treated as benign in RanSAP (may include BitLocker)
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
RANSAP_FEAT   = Path("/scratch/aabdulj4/RanSAP/features")
RANSMAP_FEAT  = Path("/scratch/aabdulj4/RanSMAP/features")   # RanSMAP features dir
CHECKPOINT    = RANSMAP_FEAT / "classifier_2way.pt"
LOG_PATH      = Path("/scratch/aabdulj4/logs/ransap_eval.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_WINDOWS   = 300
EMB_DIM     = 384     # MiniLM
FEAT_DIM    = 5       # chi_s only
SEEDS       = [42, 123, 777]
EPOCHS_CNN  = 80
EPOCHS_LSTM = 50
BATCH_SIZE  = 32
LR          = 1e-3

# Binary-fusion thresholds (same as main pipeline)
CNN_THRESH    = 0.5
STREAM_THRESH = 0.65
FPR_TARGET    = 0.05   # target FPR for lstm threshold calibration

# RanSAP label conventions (confirmed from actual directory listing)
# Overlaps with RanSMAP: Darkside, Ryuk, Sodinokibi, WannaCry
# Novel to RanSAP:       AESCrypt, Cerber, GandCrab4, TeslaCrypt
# RanSAP official composition: 7 ransomware + 5 benign (AESCrypt is BENIGN/LegitEnc)
RAN_LABELS    = ["Cerber", "Darkside", "GandCrab4",
                 "Ryuk", "Sodinokibi", "TeslaCrypt", "WannaCry"]
# Benign: standard apps + AESCrypt (legit encryption) + SDelete + Zip (all high-entropy by design)
BENIGN_LABELS = ["AESCrypt", "Excel", "Firefox", "SDelete", "Zip"]
SDELETE_LABEL = "SDelete"   # track separately for honest FPR split
BITLOCKER_LABELS = []        # not present in this dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Device: {device}")

# ── Model definitions (must match binary_fusion.py / full_lofo_final.py) ──────
RAN_IDX = 1  # class index for ransomware in classifier_2way.pt

class TextEmbeddingClassifier(nn.Module):
    """Exact architecture used to train classifier_2way.pt (from binary_fusion.py)."""
    def __init__(self, emb_dim=384, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.classifier    = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                           nn.Dropout(0.3), nn.Linear(32, n_classes))
        self.reconstructor = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),
                                           nn.Linear(128, emb_dim))
    def forward(self, x):
        feat   = self.conv(x.permute(0, 2, 1))
        pooled = feat.mean(dim=-1)
        return self.classifier(pooled), self.reconstructor(pooled), pooled

class BiLSTM(nn.Module):
    """Storage-only BiLSTM anomaly head (5-dim input)."""
    def __init__(self, input_dim=5, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                # (B, T, 2H)
        out2   = out.permute(0, 2, 1)        # (B, 2H, T)
        g      = self.pool(out2).squeeze(-1) # (B, 2H)
        score  = torch.sigmoid(self.fc(g)).squeeze(-1)
        return score

def train_bilstm(X_ran, X_ben, seeds=SEEDS, epochs=EPOCHS_LSTM, scaler=None):
    """Train storage-only BiLSTM on RanSMAP; returns model + lstm_thresh."""
    # X_ran: (n_ran, 300, 5), X_ben: (n_ben, 300, 5)
    # Scale per-feature using scaler (fitted on training set)
    n_ran, n_ben = len(X_ran), len(X_ben)
    y_ran = np.ones(n_ran, dtype=np.float32)
    y_ben = np.zeros(n_ben, dtype=np.float32)
    X_all = np.concatenate([X_ran, X_ben], axis=0)
    y_all = np.concatenate([y_ran, y_ben])

    if scaler is None:
        scaler = StandardScaler()
        X_flat = X_all.reshape(-1, FEAT_DIM)
        scaler.fit(X_flat)
    X_scaled = (X_all.reshape(-1, FEAT_DIM) - scaler.mean_) / (scaler.scale_ + 1e-8)
    X_scaled = X_scaled.reshape(-1, N_WINDOWS, FEAT_DIM)

    models = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = BiLSTM(input_dim=FEAT_DIM).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        Xt    = torch.tensor(X_scaled, dtype=torch.float32)
        yt    = torch.tensor(y_all,    dtype=torch.float32)
        ds    = TensorDataset(Xt, yt)
        dl    = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        pos_w = (n_ben / n_ran) if n_ran > 0 else 1.0
        crit  = nn.BCELoss(reduction='none')
        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                score  = model(xb)
                w      = torch.where(yb > 0.5, torch.tensor(pos_w, device=device),
                                     torch.tensor(1.0, device=device))
                loss   = (crit(score, yb) * w).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        models.append(model)

    # Calibrate lstm_thresh on training benign set (target FPR_TARGET)
    # Use the mean ensemble score over val benign
    model_scores = []
    for model in models:
        model.eval()
        with torch.no_grad():
            Xt_b = torch.tensor(X_scaled[len(X_ran):], dtype=torch.float32).to(device)
            sc   = model(Xt_b).cpu().numpy()
        model_scores.append(sc)
    ben_scores = np.stack(model_scores).mean(axis=0)
    lstm_thresh = float(np.percentile(ben_scores, 100 * (1 - FPR_TARGET)))
    log(f"    lstm_thresh calibrated to {lstm_thresh:.4f} at target FPR={FPR_TARGET:.2f}")
    return models, scaler, lstm_thresh

def ensemble_lstm_score(models, X_scaled):
    scores = []
    for model in models:
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            sc = model(Xt).cpu().numpy()
        scores.append(sc)
    return np.stack(scores).mean(axis=0)

# ── Load data ─────────────────────────────────────────────────────────────────
log("=" * 70)
log("Loading data")
log("=" * 70)

# RanSAP
rsp_feat5     = np.load(RANSAP_FEAT / "ransap_feat5.npy")        # (N, 300, 5)
rsp_text_seqs = np.load(RANSAP_FEAT / "ransap_text_seqs.npy")    # (N, 300, 384)
rsp_labels_raw = np.load(RANSAP_FEAT / "ransap_labels.npy",
                         allow_pickle=True).astype(str)

# Normalize variant suffixes: -w10dirs-02..17, -w10dirs, -largefiles
import re
def norm_label(lbl):
    return re.sub(r'-(largefiles|w10dirs.*)$', '', lbl)
rsp_labels = np.array([norm_label(l) for l in rsp_labels_raw])

log(f"  RanSAP feat5:     {rsp_feat5.shape}")
log(f"  RanSAP text_seqs: {rsp_text_seqs.shape}")
log(f"  RanSAP labels (normalized): {np.unique(rsp_labels).tolist()}")
log("  Label counts after normalization:")
for lbl in sorted(np.unique(rsp_labels)):
    log(f"    {lbl}: {(rsp_labels == lbl).sum()}")

# Assign binary role
rsp_role = np.full(len(rsp_labels), "unknown", dtype=object)
for i, lbl in enumerate(rsp_labels):
    if lbl in RAN_LABELS:            rsp_role[i] = "ransomware"
    elif lbl in BENIGN_LABELS:       rsp_role[i] = "benign"
    elif lbl in BITLOCKER_LABELS:    rsp_role[i] = "bitlocker"
    else:
        rsp_role[i] = "unknown"
        log(f"  WARNING: unrecognized label '{lbl}' — will be skipped in eval")

log("  RanSAP role counts:")
for role in ["ransomware", "benign", "bitlocker", "unknown"]:
    cnt = (rsp_role == role).sum()
    log(f"    {role}: {cnt}")

# RanSMAP storage features (for BiLSTM training)
# feat_23 is stored per-execution; load via metadata.csv filepath column
log()
smap_meta   = pd.read_csv(RANSMAP_FEAT / "metadata.csv")
smap_labels = np.load(RANSMAP_FEAT / "text_labels.npy", allow_pickle=True).astype(str)
log(f"  RanSMAP metadata: {len(smap_meta)} execs")
log(f"  RanSMAP labels:   {np.unique(smap_labels).tolist()}")

# Use only training split execs (not val) for training BiLSTM
split      = np.load(RANSMAP_FEAT / "fewshot_split_cnn.npz", allow_pickle=True)
train_idx  = split["train_idx"]   # indices into text_labels / metadata
val_idx    = split["val_idx"]

log(f"  RanSMAP train: {len(train_idx)}, val: {len(val_idx)}")
log("  Loading RanSMAP feat_23 (per-exec .npy files)...")
t_load = time.time()
smap_feat23 = np.zeros((len(smap_meta), N_WINDOWS, 23), dtype=np.float32)
for ii, row in smap_meta.iterrows():
    fp = row.get("filepath", row.get("feat_path", None))
    if fp and Path(fp).exists():
        smap_feat23[ii] = np.load(fp)
log(f"  feat_23 loaded: {smap_feat23.shape}  ({time.time()-t_load:.1f}s)")
log(f"  RanSMAP labels:  {np.unique(smap_labels).tolist()}")

# Storage features = first 5 columns
smap_feat5  = smap_feat23[:, :, :5]   # (N, 300, 5)

# RanSMAP class definitions (from binary_fusion.py)
RANSMAP_RANSOMWARE = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
RANSMAP_NONRAN     = ["AESCrypt", "SDelete", "Zip",      # LEnc (legitimate encryption)
                      "Firefox",  "Office",  "Idle"]     # Benign

def is_smap_ran(lbl):
    # Handles Conti_01..07 and mixed variants (AESCrypt_Conti etc.)
    return any(lbl == r or lbl.startswith(r + "_") or lbl.startswith(r + "-")
               for r in RANSMAP_RANSOMWARE)

def is_smap_ben(lbl):
    return any(lbl == b or lbl.startswith(b + "_") or lbl.startswith(b + "-")
               for b in RANSMAP_NONRAN)

# Use only training split execs
train_ran_smap_idx = np.array([i for i in train_idx if is_smap_ran(smap_labels[i])])
train_ben_smap_idx = np.array([i for i in train_idx if is_smap_ben(smap_labels[i])])

log(f"  RanSMAP train ran: {len(train_ran_smap_idx)}, train ben: {len(train_ben_smap_idx)}")

# Load MiniLM CNN (trained on RanSMAP, frozen for cross-dataset)
log()
log("  Loading classifier_2way.pt...")
cnn_clf = TextEmbeddingClassifier(emb_dim=EMB_DIM, n_classes=2).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device)
# Handle possible nested state dict
if "model_state_dict" in ckpt:
    cnn_clf.load_state_dict(ckpt["model_state_dict"])
elif "state_dict" in ckpt:
    cnn_clf.load_state_dict(ckpt["state_dict"])
else:
    cnn_clf.load_state_dict(ckpt)
cnn_clf.eval()
log("  classifier_2way.pt loaded")

# ── Streaming inference (zero-padded, matches MiniLM streaming in main pipeline) ──
W = 50

@torch.no_grad()
def streaming_scores(text_seqs, indices):
    """Returns streaming prob for each exec in indices."""
    scores = []
    for i in indices:
        seq  = text_seqs[int(i)]  # (300, 384)
        best = 0.0
        for start in range(0, N_WINDOWS - W + 1, 5):
            chunk  = seq[start:start + W]
            padded = np.concatenate([chunk, np.zeros((N_WINDOWS - W, EMB_DIM), np.float32)])
            x      = torch.tensor(padded[None], dtype=torch.float32).to(device)
            logits, _, _ = cnn_clf(x)
            p = torch.softmax(logits, dim=1)[0, RAN_IDX].item()
            if p > best: best = p
        scores.append(best)
    return np.array(scores, dtype=np.float32)

@torch.no_grad()
def cnn_scores(text_seqs, indices, batch=64):
    """Returns full-sequence CNN prob for each exec in indices."""
    all_scores = []
    idx_list = list(indices)
    for b in range(0, len(idx_list), batch):
        batch_idx = idx_list[b:b+batch]
        X = np.stack([text_seqs[i] for i in batch_idx])
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        logits, _, _ = cnn_clf(Xt)
        probs = torch.softmax(logits, dim=1)[:, RAN_IDX].cpu().numpy()
        all_scores.append(probs)
    return np.concatenate(all_scores)

def binary_verdict(p_ran, s, lstm_score, lstm_thresh):
    if p_ran >= CNN_THRESH:             return "Attack", "classifier"
    if s >= STREAM_THRESH:              return "Attack", "streaming"
    if lstm_score > lstm_thresh:        return "Attack", "bilstm"
    return "Benign", "none"

def report_results(verdicts, drivers, true_roles, labels, title):
    """Print per-family recall and per-benign-family FPR breakdown."""
    log(); log(f"  {title}")
    log(f"  {'Family':<20} {'N':>5} {'Detected':>9} {'Recall':>8} {'Drivers'}")
    log("  " + "-" * 70)

    ran_idx = np.where((true_roles == "ransomware") | (true_roles == "bitlocker"))[0]
    ben_idx = np.where(true_roles == "benign")[0]

    overall_tp = sum(verdicts[i] == "Attack" for i in ran_idx)
    overall_fp = sum(verdicts[i] == "Attack" for i in ben_idx)
    fpr        = overall_fp / len(ben_idx) if len(ben_idx) > 0 else float("nan")

    # Per-ransomware-family recall
    for fam in sorted(np.unique(labels[ran_idx])):
        fidx   = np.where(labels == fam)[0]
        n      = len(fidx)
        n_det  = sum(verdicts[i] == "Attack" for i in fidx)
        recall = n_det / n if n > 0 else float("nan")
        drv    = {d: sum(drivers[i] == d for i in fidx if verdicts[i] == "Attack")
                  for d in ["classifier", "streaming", "bilstm"]}
        log(f"  {fam:<20} {n:>5} {n_det:>9} {recall*100:>7.1f}%  {drv}")

    log("  " + "-" * 70)
    log(f"  {'TOTAL RAN':<20} {len(ran_idx):>5} {overall_tp:>9} "
        f"{100*overall_tp/len(ran_idx):.1f}%")
    log()

    # Per-benign-family FP breakdown
    log(f"  FPR breakdown (benign families):")
    log(f"  {'Family':<20} {'N':>5} {'FP':>5} {'FPR':>8}")
    log("  " + "-" * 42)
    for fam in sorted(np.unique(labels[ben_idx])):
        fidx = np.where(labels == fam)[0]
        n    = len(fidx)
        fp   = sum(verdicts[i] == "Attack" for i in fidx)
        note = " ← legit encryption tool (high-entropy)"  if fam == "AESCrypt" else \
               " ← secure-delete (high-entropy writes)"   if fam == "SDelete"  else \
               " ← compression (high-entropy output)"     if fam == "Zip"      else ""
        log(f"  {fam:<20} {n:>5} {fp:>5} {100*fp/n:>7.1f}%{note}")
    log("  " + "-" * 42)
    log(f"  {'TOTAL BENIGN':<20} {len(ben_idx):>5} {overall_fp:>5} {100*fpr:>7.1f}%")
    # FPR on standard benign only (excl. high-entropy tools: AESCrypt, SDelete, Zip)
    HIGH_ENT = {"AESCrypt", "SDelete", "Zip"}
    std_ben_idx = np.where((true_roles == "benign") & ~np.isin(labels, list(HIGH_ENT)))[0]
    fp_std      = sum(verdicts[i] == "Attack" for i in std_ben_idx)
    fpr_std     = fp_std / len(std_ben_idx) if len(std_ben_idx) > 0 else float("nan")
    log(f"  {'FPR std benign only':<20} {len(std_ben_idx):>5} {fp_std:>5} {100*fpr_std:>7.1f}%  (excl. AESCrypt/SDelete/Zip)")

    return overall_tp, overall_fp, fpr

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Cross-dataset transfer (RanSMAP → RanSAP)
# ═══════════════════════════════════════════════════════════════════════════════
log()
log("=" * 70)
log("EXPERIMENT 1: Cross-dataset transfer (RanSMAP → RanSAP)")
log("=" * 70)
log()

# Train storage-only BiLSTM on RanSMAP training split
log(f"Training storage-only BiLSTM on RanSMAP "
    f"({len(train_ran_smap_idx)} ran + {len(train_ben_smap_idx)} non-ran)...")
t0 = time.time()
lstm_models, scaler, lstm_thresh = train_bilstm(
    X_ran = smap_feat5[train_ran_smap_idx],
    X_ben = smap_feat5[train_ben_smap_idx]
)
log(f"  Training done in {time.time()-t0:.0f}s")

# Score all RanSAP execs
rsp_active = np.where(rsp_role != "unknown")[0]
rsp_active_feat5 = rsp_feat5[rsp_active]
rsp_active_flat  = rsp_active_feat5.reshape(-1, FEAT_DIM)
rsp_active_scaled = (rsp_active_flat - scaler.mean_) / (scaler.scale_ + 1e-8)
rsp_active_scaled = rsp_active_scaled.reshape(-1, N_WINDOWS, FEAT_DIM)

log(f"  Scoring {len(rsp_active)} RanSAP execs...")
t0 = time.time()
p_cnn    = cnn_scores(rsp_text_seqs, rsp_active)
s_stream = streaming_scores(rsp_text_seqs, rsp_active)
lstm_sc  = ensemble_lstm_score(lstm_models, rsp_active_scaled)
log(f"  Scoring done in {time.time()-t0:.0f}s")

verdicts1 = {}
drivers1  = {}
for ii, idx in enumerate(rsp_active):
    v, d = binary_verdict(p_cnn[ii], s_stream[ii], lstm_sc[ii], lstm_thresh)
    verdicts1[idx] = v
    drivers1[idx]  = d

verdicts1_arr = np.array([verdicts1.get(i, "Benign") for i in range(len(rsp_labels))])
drivers1_arr  = np.array([drivers1.get(i, "none")    for i in range(len(rsp_labels))])

tp1, fp1, fpr1 = report_results(verdicts1_arr, drivers1_arr,
                                  rsp_role, rsp_labels, "Cross-dataset Results")

# Experiment 2 (LOFO within RanSAP) removed — professor did not request it.
# Only Experiment 1 (cross-dataset transfer) is reported.

# ── Save log ──────────────────────────────────────────────────────────────────
LOG_PATH.write_text("\n".join(lines))
log()
log("ransap_eval.py done.")
log(f"Log saved to {LOG_PATH}")
