"""
D3 + D4 + D5 — feat_23 Discriminative / Distance Anomaly Head
==============================================================
Per HANDOFF update 33 (Opus decision, July 2 2026).

PRIMARY QUESTION: Can a head trained ONLY on known classes (no Conti_06)
exploit the feat_23 signal that the D2 oracle confirmed exists?

Approaches (priority order per Opus):
  (a) SUPERVISED (known-classes-only):
        A1 — Binary logistic regression on feat_23 stats (69-d)
        A2 — Supervised BiLSTM classifier on feat_23 sequences (300 × 23)
  (b) DISTANCE:
        B1 — Mahalanobis distance from non-ransomware training distribution
        B2 — KNN ransomware-fraction score (k=5 nearest training neighbors)
  D4 fold-in:
        Audit feat_23 feature scales + flat VAE normalization

Evaluation (for every approach):
  • Threshold calibrated via 5-fold OOF on training non-ransomware → FPR target ≤ 5%
  • Blind eval on ALL 70 Conti variants (7 families × 10 execs)
  • FPR measured on val non-ransomware

Fusion integration (D5):
  fuse_v6 = same logic as fuse_v4 but feat23_head fires instead of flat VAE MSE.
  Test A1, A2, B1, B2 heads in fusion context. Keep fuse_v4 as reference.

★  CORE RULE: known-classes-only training; blind eval; FPR always reported.
   D2 supervised oracle is NOT reproduced here — that was diagnostic only.
   Here Conti_06 is NEVER in training.

Saves:
  diag_d3_feat23_head.txt  — full report
  diag_d3_feat23_head.json — structured results
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

FEATURES_DIR   = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_TARGET_PCT = 95.0   # threshold = 95th pct of training non-ransomware OOF scores
K_FOLDS        = 5
STREAM_THRESH  = 0.65
BENIGN_IDX, LENC_IDX, RAN_IDX = 0, 1, 2

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(s)

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        return super().default(o)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Load data
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72)
log("Loading data...")
log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
text_seqs  = np.load(FEATURES_DIR / "text_sequences.npy")   # (1970, 300, 384)
meta_df    = pd.read_csv(FEATURES_DIR / "metadata.csv")

feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx     = np.where(split["mix_mask"])[0]

# Training pool: all known-class executions (1,800, includes val)
train_mask = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx  = np.where(train_mask)[0]

# Val non-ransomware (for FPR measurement and threshold calibration cross-check)
val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])

# Training labels: 0=benign, 1=legit-enc, 2=ransomware, binary=ransomware(1) vs rest(0)
def label_to_3class(lbl):
    if lbl in RANSOMWARE_CLASSES: return 2
    if lbl in LEGIT_ENC_CLASSES:  return 1
    return 0

train_y3    = np.array([label_to_3class(all_labels[i]) for i in train_idx])
train_y_bin = (train_y3 == 2).astype(int)  # binary: ransomware=1, rest=0

log(f"  Training pool : {len(train_idx)}  (ransomware: {train_y_bin.sum()}, rest: {(train_y_bin==0).sum()})")
log(f"  Val indices   : {len(val_idx)}  (non-ransomware: {len(val_nonran_idx)})")
log(f"  Variant idx   : {len(variant_idx)}  (split into 7 families)")
log(f"  Mix idx       : {len(mix_idx)}")
log(f"  Total         : {len(all_labels)}")

# ── Compute feat_23 stats (mean + std + max over 300 windows → 69-d) ──────────
feat_23_f64  = feat_23.astype(np.float64)
feat23_mean  = feat_23_f64.mean(axis=1).astype(np.float32)  # (N, 23)
feat23_std   = feat_23_f64.std(axis=1).astype(np.float32)   # (N, 23)
feat23_max   = feat_23_f64.max(axis=1).astype(np.float32)   # (N, 23)
feat23_stats = np.concatenate([feat23_mean, feat23_std, feat23_max], axis=1)  # (N, 69)
log(f"  feat_23 stats shape: {feat23_stats.shape}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — D4: Feature scale audit
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("D4 — FEATURE SCALE AUDIT")
log("=" * 72)

# Per-feature stats across training pool in ORIGINAL space
f23_tr = feat_23[train_idx]  # (1800, 300, 23) — float32
y_bin_tr = train_y_bin       # (1800,)

# Flatten to (N*300, 23) for per-feature stats
f23_flat   = f23_tr.reshape(-1, 23).astype(np.float64)
feat_means = f23_flat.mean(axis=0)
feat_stds  = f23_flat.std(axis=0)
feat_maxs  = f23_flat.max(axis=0)

log(f"\n  Per-feature statistics across training pool (N=1800, 300 windows each):")
log(f"  {'Feat':>5}  {'Mean':>12}  {'Std':>12}  {'Max':>12}  {'Std/Mean':>10}")
log(f"  {'─'*58}")
for j in range(23):
    ratio = feat_stds[j] / (abs(feat_means[j]) + 1e-8)
    log(f"  {j:>5}  {feat_means[j]:>12.3f}  {feat_stds[j]:>12.3f}  {feat_maxs[j]:>12.3f}  {ratio:>10.3f}")

# Check StandardScaler effect
sc_audit = StandardScaler()
f23_flat_scaled = sc_audit.fit_transform(f23_flat)
log(f"\n  After StandardScaler (expected: mean≈0, std≈1 per feature):")
log(f"  mean range: [{f23_flat_scaled.mean(axis=0).min():.3f}, {f23_flat_scaled.mean(axis=0).max():.3f}]  "
    f"std range: [{f23_flat_scaled.std(axis=0).min():.3f}, {f23_flat_scaled.std(axis=0).max():.3f}]")
log(f"  Normalization looks {'OK' if f23_flat_scaled.std(axis=0).max() < 1.1 else 'SUSPECT'}.")

# MSE distribution for existing flat VAE (load it for D4 comparison)
log("\n  [D4] Checking flat VAE MSE distribution across classes...")
try:
    class TimeSeriesVAE(nn.Module):
        def __init__(self, seq_len=300, n_features=23, latent_dim=16):
            super().__init__()
            D = seq_len * n_features
            self.enc = nn.Sequential(
                nn.Linear(D, 512),   nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            )
            self.fc_mu     = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 512),        nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, D),
            )
        def encode(self, x):
            h = self.enc(x.view(x.size(0), -1))
            return self.fc_mu(h), self.fc_logvar(h)
        def decode(self, z):
            return self.dec(z).view(z.size(0), 300, 23)

    vae = TimeSeriesVAE().to(device)
    vae.load_state_dict(torch.load(FEATURES_DIR / "vae_model.pt", map_location=device, weights_only=True))
    vae.eval()
    vae_mean_sc  = np.load(FEATURES_DIR / "vae_scaler_mean.npy")
    vae_scale_sc = np.load(FEATURES_DIR / "vae_scaler_scale.npy")
    vae_threshold = float(np.load(FEATURES_DIR / "vae_threshold.npy")[0])

    def vae_mse(idx):
        feat_flat   = feat_23[idx].reshape(-1, 23)
        feat_scaled = ((feat_flat - vae_mean_sc) / vae_scale_sc).reshape(1, 300, 23).astype(np.float32)
        x = torch.tensor(feat_scaled).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            recon  = vae.decode(mu)
            return F.mse_loss(recon, x, reduction="mean").item()

    # MSE distribution per class in training pool
    log(f"  VAE threshold: {vae_threshold:.4f}")
    for class_name, class_list in [("Benign", BENIGN_CLASSES), ("Legit-Enc", LEGIT_ENC_CLASSES), ("Ransomware", RANSOMWARE_CLASSES)]:
        idx_c = [i for i in train_idx if all_labels[i] in class_list][:30]  # sample 30
        mse_vals = [vae_mse(i) for i in idx_c]
        log(f"  MSE {class_name:<12}: mean={np.mean(mse_vals):.4f}  std={np.std(mse_vals):.4f}  "
            f"max={np.max(mse_vals):.4f}  pct>thresh={np.mean(np.array(mse_vals)>vae_threshold):.0%}")
    HAVE_VAE = True
except Exception as e:
    log(f"  Could not load flat VAE: {e}  (D4 VAE audit skipped)")
    HAVE_VAE = False
    vae_threshold = 0.6720
    def vae_mse(idx): return 0.0


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Load CNN classifier for fusion context
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("Loading CNN classifier for fusion context...")
log("=" * 72)

class TextEmbeddingClassifier(nn.Module):
    def __init__(self, seq_len=300, emb_dim=384, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64,  kernel_size=5, padding=2),
            nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_classes),
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, emb_dim),
        )
    def forward(self, x):
        feat   = self.conv(x.permute(0, 2, 1))
        pooled = feat.mean(dim=-1)
        return self.classifier(pooled), self.reconstructor(pooled), pooled

clf = TextEmbeddingClassifier().to(device)
clf.load_state_dict(torch.load(FEATURES_DIR / "classifier.pt", map_location=device, weights_only=True))
clf.eval()

def primary_probs(idx):
    x = torch.tensor(text_seqs[idx], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = clf(x)
        return torch.softmax(logits, dim=1).cpu().numpy()[0]

def streaming_max(idx):
    seq = text_seqs[idx]; W = 50; best = 0.0
    with torch.no_grad():
        for start in range(0, 300 - W + 1, 5):
            chunk  = seq[start:start + W]
            padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
            x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
            logits, _, _ = clf(x)
            p = torch.softmax(logits, dim=1)[0, RAN_IDX].item()
            if p > best: best = p
    return best

eval_idx  = np.concatenate([val_idx, variant_idx, mix_idx])
log(f"Precomputing primary probs for {len(eval_idx)} executions...")
pmap = {int(i): primary_probs(i) for i in eval_idx}

log("Computing streaming scores for non-Ransomware primaries...")
smap, vmap = {}, {}
for i in eval_idx:
    ii = int(i)
    if pmap[ii].argmax() != RAN_IDX:
        smap[ii] = streaming_max(i)
        vmap[ii] = vae_mse(i) if HAVE_VAE else 0.0
    else:
        smap[ii] = 0.0; vmap[ii] = 0.0
log("Primary probs + streaming done.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — B1/B2: Distance heads (Mahalanobis, KNN)
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("B1 + B2 — Distance Heads (Mahalanobis, KNN)")
log("=" * 72)

# Fit on training non-ransomware (benign + legit-enc) for Mahalanobis
train_nonran_mask = (train_y_bin == 0)
X_tr_stats        = feat23_stats[train_idx]
X_tr_nr           = X_tr_stats[train_nonran_mask]   # non-ransomware training

# StandardScaler on all training data (same scaler for all heads)
sc_global = StandardScaler()
X_tr_scaled = sc_global.fit_transform(X_tr_stats)
X_all_scaled = sc_global.transform(feat23_stats)    # (1970, 69) — scaled
X_tr_nr_scaled = X_all_scaled[train_idx][train_nonran_mask]

# ── B1: Mahalanobis from non-ransomware centroid ──────────────────────────────
log("\n[B1] Mahalanobis distance from non-ransomware training distribution...")
mu_nr  = X_tr_nr_scaled.mean(axis=0)
cov_nr = np.cov(X_tr_nr_scaled.T)
try:
    VI_nr  = np.linalg.pinv(cov_nr)    # pseudo-inverse for robustness
    log(f"     Covariance matrix size: {cov_nr.shape}  (pinv computed OK)")
    def maha_score(x_scaled_1d):
        diff = x_scaled_1d - mu_nr
        return float(np.sqrt(diff @ VI_nr @ diff))
    maha_scores_all = np.array([maha_score(X_all_scaled[i]) for i in range(len(feat23_stats))])
    log(f"     Training non-ran Mahalanobis: mean={maha_scores_all[train_idx][train_nonran_mask].mean():.3f}  "
        f"95th pct={np.percentile(maha_scores_all[train_idx][train_nonran_mask], 95):.3f}")
    ran_maha = maha_scores_all[train_idx][~train_nonran_mask]
    log(f"     Training ransomware  Mahalanobis: mean={ran_maha.mean():.3f}  min={ran_maha.min():.3f}")
    HAVE_MAHA = True
except Exception as e:
    log(f"     Mahalanobis failed: {e}")
    maha_scores_all = np.zeros(len(feat23_stats))
    HAVE_MAHA = False

# ── B2: KNN ransomware fraction ───────────────────────────────────────────────
log("\n[B2] KNN ransomware-fraction score (k=5, cosine distance)...")
knn_B2 = NearestNeighbors(n_neighbors=5, metric="cosine")
knn_B2.fit(X_tr_scaled)
_, nn_indices = knn_B2.kneighbors(X_all_scaled)   # (N, 5)
knn_ran_frac = np.array([
    (train_y_bin[nn_indices[i]] == 1).mean() for i in range(len(feat23_stats))
])
log(f"     Training non-ran KNN ran-frac: mean={knn_ran_frac[train_idx][train_nonran_mask].mean():.3f}  "
    f"95th pct={np.percentile(knn_ran_frac[train_idx][train_nonran_mask], 95):.3f}")
log(f"     Training ransomware KNN ran-frac: mean={knn_ran_frac[train_idx][~train_nonran_mask].mean():.3f}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — A1: Supervised binary logistic regression (5-fold OOF CV)
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("A1 — Supervised Binary Logistic Regression on feat_23 stats (5-fold OOF)")
log("=" * 72)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
lr_oof_scores = np.zeros(len(train_idx))   # OOF P(ransomware) for training pool

for fold, (tr_f, val_f) in enumerate(skf.split(X_tr_stats, train_y_bin)):
    X_f_tr = X_tr_stats[tr_f]; y_f_tr = train_y_bin[tr_f]
    X_f_val = X_tr_stats[val_f]
    sc_f = StandardScaler(); X_f_tr_s = sc_f.fit_transform(X_f_tr)
    X_f_val_s = sc_f.transform(X_f_val)
    clf_lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs")
    clf_lr.fit(X_f_tr_s, y_f_tr)
    lr_oof_scores[val_f] = clf_lr.predict_proba(X_f_val_s)[:, 1]

# OOF threshold: 95th pct of non-ransomware OOF scores in training pool
lr_oof_nonran_scores = lr_oof_scores[train_nonran_mask]
lr_thresh_oof = float(np.percentile(lr_oof_nonran_scores, FPR_TARGET_PCT))
lr_oof_fpr    = float((lr_oof_nonran_scores > lr_thresh_oof).mean())
lr_oof_tpr    = float((lr_oof_scores[~train_nonran_mask] > lr_thresh_oof).mean())
log(f"  OOF threshold @ {FPR_TARGET_PCT}th pct non-ran: {lr_thresh_oof:.4f}")
log(f"  OOF FPR (training non-ransomware): {lr_oof_fpr:.1%}")
log(f"  OOF TPR (training ransomware):     {lr_oof_tpr:.1%}")

# Final model on all 1,800
sc_lr_final = StandardScaler()
X_tr_final_s = sc_lr_final.fit_transform(X_tr_stats)
clf_lr_final = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs")
clf_lr_final.fit(X_tr_final_s, train_y_bin)
lr_scores_all = clf_lr_final.predict_proba(sc_lr_final.transform(feat23_stats))[:, 1]  # (N,)

# Recalibrate threshold on val non-ransomware with final model
lr_thresh_val = float(np.percentile(lr_scores_all[val_nonran_idx], FPR_TARGET_PCT))
log(f"  Final model threshold (val non-ran 95th pct): {lr_thresh_val:.4f}")
log(f"  Final model val FPR:  {float((lr_scores_all[val_nonran_idx] > lr_thresh_val).mean()):.1%}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — A2: Supervised BiLSTM classifier (GPU)
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("A2 — Supervised BiLSTM Classifier on feat_23 sequences (300 × 23)")
log("=" * 72)

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_features=23, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        D = hidden_dim * 2   # bidirectional
        self.head = nn.Sequential(
            nn.Linear(D, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (B, 300, 23)
        # BN over feature dim: reshape to (B*300, 23) → BN → reshape back
        B, T, F = x.shape
        x_bn = self.bn_input(x.reshape(B * T, F)).reshape(B, T, F)
        out, (h_n, _) = self.lstm(x_bn)
        # h_n: (n_layers*2, B, hidden_dim) — take last layer both directions
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden_dim*2)
        return self.head(h_last)   # (B, 2) — logits

def normalize_sequences(X_seqs, scaler=None):
    """Per-feature StandardScaler on (N, 300, 23) → fit or transform."""
    N, T, F = X_seqs.shape
    flat = X_seqs.reshape(-1, F).astype(np.float64)
    if scaler is None:
        scaler = StandardScaler()
        flat_s = scaler.fit_transform(flat).astype(np.float32)
    else:
        flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(N, T, F), scaler

def train_lstm(X_tr_seqs, y_tr, n_epochs=60, batch_size=64, lr=1e-3, seed=42):
    """Train BiLSTM; returns model + losses."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = BiLSTMClassifier().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Class weights: inverse frequency
    n_ran  = y_tr.sum(); n_rest = len(y_tr) - n_ran
    w = torch.tensor([n_ran / len(y_tr), n_rest / len(y_tr)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    X_t = torch.tensor(X_tr_seqs, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.long)
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    losses = []
    for ep in range(n_epochs):
        ep_loss = 0.0
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(dl))
        if (ep + 1) % 20 == 0:
            print(f"    Epoch {ep+1:3d}/{n_epochs}: loss={losses[-1]:.4f}", flush=True)
    return model, losses

@torch.no_grad()
def score_lstm(model, X_seqs):
    """P(ransomware) for all N sequences. Returns (N,) numpy array."""
    model.eval()
    X_t = torch.tensor(X_seqs, dtype=torch.float32)
    dl  = DataLoader(TensorDataset(X_t), batch_size=128, shuffle=False)
    probs = []
    for (bx,) in dl:
        bx = bx.to(device)
        logits = model(bx)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

# 5-fold OOF CV for threshold calibration
log(f"\n  5-fold OOF CV (known-classes-only, {K_FOLDS} folds × 1 seed):")
lstm_oof_scores = np.zeros(len(train_idx))
X_tr_seqs = feat_23[train_idx]  # (1800, 300, 23)

for fold, (tr_f, val_f) in enumerate(skf.split(X_tr_seqs, train_y_bin)):
    log(f"  --- Fold {fold+1}/{K_FOLDS} ---")
    X_f_tr_raw = X_tr_seqs[tr_f]; X_f_val_raw = X_tr_seqs[val_f]
    y_f_tr     = train_y_bin[tr_f]

    X_f_tr_n, scaler_f = normalize_sequences(X_f_tr_raw)
    X_f_val_n, _       = normalize_sequences(X_f_val_raw, scaler_f)

    model_f, _ = train_lstm(X_f_tr_n, y_f_tr, n_epochs=60, seed=fold)
    lstm_oof_scores[val_f] = score_lstm(model_f, X_f_val_n)
    del model_f; torch.cuda.empty_cache()

lstm_oof_nonran_scores = lstm_oof_scores[train_nonran_mask]
lstm_thresh_oof = float(np.percentile(lstm_oof_nonran_scores, FPR_TARGET_PCT))
lstm_oof_fpr    = float((lstm_oof_nonran_scores > lstm_thresh_oof).mean())
lstm_oof_tpr    = float((lstm_oof_scores[~train_nonran_mask] > lstm_thresh_oof).mean())
log(f"\n  LSTM OOF threshold @ {FPR_TARGET_PCT}th pct non-ran: {lstm_thresh_oof:.4f}")
log(f"  LSTM OOF FPR (training non-ransomware): {lstm_oof_fpr:.1%}")
log(f"  LSTM OOF TPR (training ransomware):     {lstm_oof_tpr:.1%}")

# Final LSTM: 3 seeds, average P(ransomware)
log("\n  Training final BiLSTM (3 seeds on all 1,800)...")
X_all_seqs_n, sc_lstm_final = normalize_sequences(feat_23[train_idx])

final_lstm_models = []
for seed in [0, 1, 2]:
    log(f"  Seed {seed}:")
    model_s, _ = train_lstm(X_all_seqs_n, train_y_bin, n_epochs=80, seed=seed)
    final_lstm_models.append(model_s)

# Score ALL executions with final models (ensemble average)
X_all_norm, _ = normalize_sequences(feat_23, sc_lstm_final)
lstm_scores_ensemble = np.mean([score_lstm(m, X_all_norm) for m in final_lstm_models], axis=0)

lstm_thresh_val = float(np.percentile(lstm_scores_ensemble[val_nonran_idx], FPR_TARGET_PCT))
log(f"  Final LSTM threshold (val non-ran 95th pct): {lstm_thresh_val:.4f}")
log(f"  Final LSTM val FPR:  {float((lstm_scores_ensemble[val_nonran_idx] > lstm_thresh_val).mean()):.1%}")
log(f"  Final LSTM val TPR (known ransomware): {float((lstm_scores_ensemble[[i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES]] > lstm_thresh_val).mean()):.1%}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Threshold calibration for distance heads
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("Calibrating thresholds for distance heads (B1, B2) on val non-ransomware")
log("=" * 72)

# B1 Mahalanobis
if HAVE_MAHA:
    maha_thresh = float(np.percentile(maha_scores_all[val_nonran_idx], FPR_TARGET_PCT))
    maha_val_fpr = float((maha_scores_all[val_nonran_idx] > maha_thresh).mean())
    log(f"  B1 Mahalanobis threshold: {maha_thresh:.3f}  val FPR: {maha_val_fpr:.1%}")
else:
    maha_thresh = float("inf"); maha_val_fpr = 0.0

# B2 KNN
knn_thresh = float(np.percentile(knn_ran_frac[val_nonran_idx], FPR_TARGET_PCT))
knn_val_fpr = float((knn_ran_frac[val_nonran_idx] > knn_thresh).mean())
log(f"  B2 KNN threshold: {knn_thresh:.3f}  val FPR: {knn_val_fpr:.1%}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Fusion integration (fuse_v6 replacing flat VAE)
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("Fusion Integration — fuse_v6: feat_23 head replaces flat VAE")
log("=" * 72)

def fuse_v4_orig(p, s, v):
    """v4 original (flat VAE reference)."""
    def vae_fires(vv): return 0 < vv < 1e6 and vv > vae_threshold
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]: return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        if s >= STREAM_THRESH or vae_fires(v): return "Attack"
        return "Benign"
    if s >= STREAM_THRESH: return "Attack"
    if vae_fires(v): return "Idle"
    return "Benign"

def fuse_v6(p, s, head_score, thresh):
    """fuse_v4 logic with feat_23 head instead of flat VAE."""
    fires = head_score > thresh
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]: return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"
    if s >= STREAM_THRESH: return "Attack"
    if fires: return "Idle"
    return "Benign"

# Head score maps for all eval indices
score_maps = {
    "A1_LR":   (lr_scores_all,         lr_thresh_val),
    "A2_LSTM": (lstm_scores_ensemble,   lstm_thresh_val),
    "B1_Maha": (maha_scores_all,        maha_thresh),
    "B2_KNN":  (knn_ran_frac,           knn_thresh),
}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Blind evaluation
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("BLIND EVALUATION — All 70 Conti variants + val FPR")
log("=" * 72)

json_out = {"heads": {}, "variant_recall": {}, "val_fpr": {}}

# ── Val FPR for each fusion version ──────────────────────────────────────────
log(f"\n── VAL FPR ──")
non_ran_val = [i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES]
log(f"{'Fusion':>14}  {'Benign FPR':>12}  {'Legit-Enc FPR':>14}  {'Combined FPR':>13}")
log("─" * 58)

all_fusions = {"v4_orig": None}
for name in score_maps: all_fusions[f"v6_{name}"] = name

for fusion_name, head_name in all_fusions.items():
    ben_idx  = [i for i in val_idx if all_labels[i] in BENIGN_CLASSES]
    lenc_idx = [i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES]
    nr_idx   = non_ran_val

    def get_verdict(i, fname, hname):
        p = pmap[int(i)]; s = smap[int(i)]
        if fname == "v4_orig":
            return fuse_v4_orig(p, s, vmap[int(i)])
        scores_arr, thresh = score_maps[hname]
        return fuse_v6(p, s, scores_arr[int(i)], thresh)

    b_fp   = sum(1 for i in ben_idx  if get_verdict(i, fusion_name, head_name) == "Attack")
    le_fp  = sum(1 for i in lenc_idx if get_verdict(i, fusion_name, head_name) == "Attack")
    nr_fp  = sum(1 for i in nr_idx   if get_verdict(i, fusion_name, head_name) == "Attack")

    b_fpr  = b_fp  / max(len(ben_idx),  1)
    le_fpr = le_fp / max(len(lenc_idx), 1)
    nr_fpr = nr_fp / max(len(nr_idx),   1)

    log(f"  {fusion_name:>14}  {b_fp}/{len(ben_idx)} ({100*b_fpr:.0f}%)   "
        f"{le_fp}/{len(lenc_idx)} ({100*le_fpr:.0f}%)       "
        f"{nr_fp}/{len(nr_idx)} ({100*nr_fpr:.0f}%)")

    json_out["val_fpr"][fusion_name] = {"benign": b_fpr, "legit_enc": le_fpr, "combined": nr_fpr}

# ── Variant recall for each fusion version ────────────────────────────────────
log(f"\n── VARIANT RECALL (blind, no Conti_06 in training) ──")

for fusion_name, head_name in all_fusions.items():
    log(f"\n  [{fusion_name}]")
    log(f"  {'Variant':<14}  {'N':>4}  {'Attack':>7}  {'Idle':>6}  {'Benign':>7}  {'Recall':>8}")
    log(f"  {'─'*54}")

    total_att = 0; total_n = 0
    per_variant = {}
    for vname in VARIANT_CLASSES:
        vi = [i for i in variant_idx if all_labels[i] == vname]
        n  = len(vi)
        if n == 0: continue

        verdicts = []
        for i in vi:
            p = pmap[int(i)]; s = smap[int(i)]
            if fusion_name == "v4_orig":
                verdicts.append(fuse_v4_orig(p, s, vmap[int(i)]))
            else:
                scores_arr, thresh = score_maps[head_name]
                verdicts.append(fuse_v6(p, s, scores_arr[int(i)], thresh))

        n_att  = sum(v == "Attack" for v in verdicts)
        n_idle = sum(v == "Idle"   for v in verdicts)
        n_ben  = sum(v == "Benign" for v in verdicts)
        recall = n_att / n
        flag   = " ◀" if vname == "Conti_06" else ""
        total_att += n_att; total_n += n
        log(f"  {vname+flag:<14}  {n:>4}  {n_att:>7}  {n_idle:>6}  {n_ben:>7}  {recall:>8.0%}")
        per_variant[vname] = {"n": n, "attack": n_att, "idle": n_idle, "benign": n_ben, "recall": recall}

    overall_recall = total_att / total_n
    log(f"  {'ALL':14}  {total_n:>4}  {total_att:>7}  {'':>6}  {'':>7}  {overall_recall:>8.0%}")
    json_out["variant_recall"][fusion_name] = {"overall": overall_recall, "per_variant": per_variant}

# ── Per-execution detail for Conti_06 (all fusion versions) ──────────────────
log(f"\n── CONTI_06 PER-EXECUTION DETAIL ──")
c06_idx_all = [i for i in variant_idx if all_labels[i] == "Conti_06"]

header_parts = [f"{'ExecIdx':>8}"]
for fname in all_fusions: header_parts.append(f"{fname:>14}")
log("  " + "  ".join(header_parts))
log("  " + "─" * (8 + 16 * len(all_fusions)))

for i in c06_idx_all:
    row = [f"{i:>8}"]
    for fname, hname in all_fusions.items():
        p = pmap[int(i)]; s = smap[int(i)]
        if fname == "v4_orig":
            v = fuse_v4_orig(p, s, vmap[int(i)])
        else:
            scores_arr, thresh = score_maps[hname]
            v = fuse_v6(p, s, scores_arr[int(i)], thresh)
        score_str = ""
        if hname:
            score_str = f"{score_maps[hname][0][int(i)]:.3f}"
        row.append(f"{v:>8}/{score_str:>5}")
    log("  " + "  ".join(row))


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Summary + recommendation
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("SUMMARY — D3+D4+D5 feat_23 Discriminative Head")
log("=" * 72)

log(f"\n  Reference: fusion v4 (flat VAE)  →  84% variant recall / ~1.7% combined FPR")
log(f"  {'Head':>14}  {'Overall Recall':>15}  {'Val FPR':>9}  {'Conti_06 recall':>16}")
log(f"  {'─'*58}")

for fname, hname in all_fusions.items():
    ov_rec  = json_out["variant_recall"][fname]["overall"]
    nr_fpr  = json_out["val_fpr"][fname]["combined"]
    c06_rec = json_out["variant_recall"][fname]["per_variant"].get("Conti_06", {}).get("recall", 0.0)
    log(f"  {fname:>14}  {ov_rec:>15.0%}  {nr_fpr:>9.1%}  {c06_rec:>16.0%}")

log()
log("  Interpretation guide:")
log("  ▸ A head that IMPROVES overall recall AND maintains val FPR ≈ 1-2% is a win.")
log("  ▸ A head that improves Conti_06 recall without hurting other families is a partial win.")
log("  ▸ Distance heads (B1/B2) act unsupervised — lower FPR target than supervised.")
log("  ▸ If A2_LSTM >> A1_LR: the temporal trajectory (not just stats) carries the signal.")
log("  ▸ If none improve Conti_06: Conti_06 is an information-limit finding.")


# ── Save outputs ─────────────────────────────────────────────────────────────
json_out["thresholds"] = {
    "A1_LR":   {"val": lr_thresh_val, "oof": lr_thresh_oof},
    "A2_LSTM": {"val": lstm_thresh_val, "oof": lstm_thresh_oof},
    "B1_Maha": {"val": maha_thresh},
    "B2_KNN":  {"val": knn_thresh},
}
json_out["oof_metrics"] = {
    "A1_LR":   {"fpr": lr_oof_fpr,   "tpr": lr_oof_tpr},
    "A2_LSTM": {"fpr": lstm_oof_fpr, "tpr": lstm_oof_tpr},
}

summary_txt = "\n".join(lines)
(FEATURES_DIR / "diag_d3_feat23_head.txt").write_text(summary_txt)
with open(FEATURES_DIR / "diag_d3_feat23_head.json", "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)

print(f"\nSaved: diag_d3_feat23_head.txt / .json")
print("D3 done.")
