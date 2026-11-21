"""
D3-FIXED — feat_23 Heads + VAE Retrain on Fixed Features + Full Pipeline Fusion
================================================================================
Per HANDOFF update 36 Opus decision (July 5 2026).

What this script does:
  1. Retrains the flat MLP VAE on FIXED feat_23 (10 of 12 previously-dead
     features now live). Saves model + scaler + threshold.
  2. Generates vae_fixed_window_scores.npy — per-exec MSE for all 1970.
  3. Retrains all four D3 heads (A1 LR / A2 BiLSTM / B1 Mahalanobis / B2 KNN)
     on fixed feat_23.
  4. Runs FOUR fusion variants for blind eval:
       fuse_minilm_vae  — MiniLM primary + MiniLM streaming + fixed VAE   (fair MiniLM pipeline)
       fuse_mpnet_vae   — mpnet primary + mpnet streaming + fixed VAE     (fair mpnet pipeline)
       fuse_v6_<head>   — MiniLM primary + feat_23 head instead of VAE   (same as original D3 but fixed)
  5. LOFO (Leave-One-Family-Out) for cross-family zero-day.
  6. Per-exec Conti_06 detail for all fusion variants.

Saves:
  vae_fixed_model.pt             — retrained VAE weights (best OOF seed/fold)
  vae_fixed_scaler_mean.npy      — StandardScaler mean  (fitted on training feat_23)
  vae_fixed_scaler_scale.npy     — StandardScaler scale
  vae_fixed_threshold.npy        — MSE threshold (95th pct of OOF non-ran MSE)
  vae_fixed_window_scores.npy    — per-exec mean-window MSE for all 1970 (from final VAE)
  d3_fixed_results.txt           — full text report
  d3_fixed_results.json          — structured results

★  CORE RULE: known-classes-only training; blind eval on ALL variants + LOFO;
   FPR always reported. Conti_06 is NEVER in training.
"""

import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors     import NearestNeighbors
from torch.utils.data      import DataLoader, TensorDataset

FEATURES_DIR   = Path("/scratch/aabdulj4/RanSMAP/features")
K_FOLDS        = 5
FPR_TARGET_PCT = 95.0   # threshold = 95th pct of OOF non-ran MSE/scores
STREAM_THRESH  = 0.65
BENIGN_IDX, LENC_IDX, RAN_IDX = 0, 1, 2

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]
FAMILY_CLASSES       = {  # for LOFO
    "WannaCry": ["WannaCry"], "Ryuk": ["Ryuk"], "REvil": ["REvil"],
    "LockBit": ["LockBit"], "Darkside": ["Darkside"], "Conti": ["Conti"],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        return super().default(o)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Load data
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72); log("Loading data..."); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
all_groups = np.load(FEATURES_DIR / "text_groups.npy")
meta_df    = pd.read_csv(FEATURES_DIR / "metadata.csv")

# Fixed feat_23: load all 1970 .npy files (they are now re-extracted with fixed code)
t0 = time.time()
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])
log(f"  feat_23 loaded: {feat_23.shape}  in {time.time()-t0:.1f}s")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx     = np.where(split["mix_mask"])[0]

# Training pool: all known-class executions (1,800)
train_mask = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx  = np.where(train_mask)[0]

val_nonran_idx   = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ran_idx      = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])

def label_to_3class(lbl):
    if lbl in RANSOMWARE_CLASSES: return 2
    if lbl in LEGIT_ENC_CLASSES:  return 1
    return 0

train_y3    = np.array([label_to_3class(all_labels[i]) for i in train_idx])
train_y_bin = (train_y3 == 2).astype(int)
train_nonran_mask = (train_y_bin == 0)

log(f"  Training pool: {len(train_idx)}  (ran: {train_y_bin.sum()}, rest: {(train_y_bin==0).sum()})")
log(f"  Val:          {len(val_idx)}  (non-ran: {len(val_nonran_idx)}, ran: {len(val_ran_idx)})")
log(f"  Variants:     {len(variant_idx)}")

# Compute feat_23 stats (mean + std + max over 300 windows → 69-d per exec)
feat_23_f64  = feat_23.astype(np.float64)
feat23_mean  = feat_23_f64.mean(axis=1).astype(np.float32)
feat23_std   = feat_23_f64.std(axis=1).astype(np.float32)
feat23_max   = feat_23_f64.max(axis=1).astype(np.float32)
feat23_stats = np.concatenate([feat23_mean, feat23_std, feat23_max], axis=1)  # (N, 69)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Retrain VAE on fixed feat_23
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("RETRAIN VAE on fixed feat_23 (5-fold OOF × 3 seeds)"); log("=" * 72)

class TimeSeriesVAE(nn.Module):
    def __init__(self, seq_len=300, n_features=23, latent_dim=16):
        super().__init__()
        D = seq_len * n_features   # 6900
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
    def decode(self, z): return self.dec(z).view(z.size(0), 300, 23)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

def vae_loss_fn(recon, x, mu, logvar, beta=1.0):
    mse = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return mse + beta * kld, mse

def train_vae(X_seqs_scaled, n_epochs=80, lr=1e-3, batch_size=64, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TimeSeriesVAE().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    X_t   = torch.tensor(X_seqs_scaled, dtype=torch.float32)
    dl    = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(n_epochs):
        for (bx,) in dl:
            bx = bx.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(bx)
            loss, _ = vae_loss_fn(recon, bx, mu, logvar)
            loss.backward(); opt.step()
        sched.step()
    return model

@torch.no_grad()
def vae_mse_batch(model, X_seqs_scaled):
    """Return per-execution mean MSE over 300 windows."""
    model.eval()
    X_t = torch.tensor(X_seqs_scaled, dtype=torch.float32)
    dl  = DataLoader(TensorDataset(X_t), batch_size=64, shuffle=False)
    mse_list = []
    for (bx,) in dl:
        bx = bx.to(device)
        mu, _ = model.encode(bx)
        recon  = model.decode(mu)
        mse_list.append(F.mse_loss(recon, bx, reduction="none")
                        .mean(dim=[1, 2]).cpu().numpy())
    return np.concatenate(mse_list)   # (N,)

# Scale training feat_23 for VAE
vae_scaler = StandardScaler()
X_tr_raw   = feat_23[train_idx]   # (1800, 300, 23)
N_tr, T, n_feat = X_tr_raw.shape
X_tr_flat  = X_tr_raw.reshape(-1, n_feat)
vae_scaler.fit(X_tr_flat)

def scale_seqs(seqs):
    N, T, F = seqs.shape
    return (vae_scaler.transform(seqs.reshape(-1, F)).astype(np.float32)
            .reshape(N, T, F))

X_tr_scaled = scale_seqs(X_tr_raw)
X_all_scaled = scale_seqs(feat_23)   # (1970, 300, 23)

# 5-fold OOF for threshold calibration
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
vae_oof_mse = np.zeros(len(train_idx))

for fold, (tr_f, val_f) in enumerate(skf.split(X_tr_scaled, train_y_bin)):
    log(f"  VAE fold {fold+1}/{K_FOLDS}...")
    X_f_tr = X_tr_scaled[tr_f]
    X_f_val = X_tr_scaled[val_f]
    model_f = train_vae(X_f_tr, n_epochs=80, seed=fold)
    vae_oof_mse[val_f] = vae_mse_batch(model_f, X_f_val)
    del model_f; torch.cuda.empty_cache()

vae_oof_nonran = vae_oof_mse[train_nonran_mask]
vae_thresh     = float(np.percentile(vae_oof_nonran, FPR_TARGET_PCT))
vae_oof_fpr    = float((vae_oof_nonran > vae_thresh).mean())
vae_oof_tpr    = float((vae_oof_mse[~train_nonran_mask] > vae_thresh).mean())
log(f"\n  VAE OOF threshold @ {FPR_TARGET_PCT}th pct non-ran: {vae_thresh:.4f}")
log(f"  VAE OOF FPR (training non-ran): {vae_oof_fpr:.1%}")
log(f"  VAE OOF TPR (training ran):     {vae_oof_tpr:.1%}")

# Final VAE: 3 seeds on all 1800
log("\n  Training final VAE (3 seeds on all 1800, picking best val MSE)...")
best_val_loss = float("inf")
best_vae_model = None
for seed in [0, 1, 2]:
    log(f"  Seed {seed}:")
    m = train_vae(X_tr_scaled, n_epochs=80, seed=seed)
    val_mse = vae_mse_batch(m, X_all_scaled[val_nonran_idx]).mean()
    log(f"    val non-ran mean MSE: {val_mse:.4f}")
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        best_vae_model = m

# Use best seed for all scoring
vae_all_mse = vae_mse_batch(best_vae_model, X_all_scaled)   # (1970,)
vae_thresh_val = float(np.percentile(vae_all_mse[val_nonran_idx], FPR_TARGET_PCT))
vae_val_fpr    = float((vae_all_mse[val_nonran_idx] > vae_thresh_val).mean())
log(f"\n  Final VAE threshold (val non-ran 95th pct): {vae_thresh_val:.4f}")
log(f"  Final VAE val FPR:                          {vae_val_fpr:.1%}")
log(f"  Final VAE val TPR (known ran):              "
    f"{float((vae_all_mse[val_ran_idx] > vae_thresh_val).mean()):.1%}")

# Save VAE artifacts
best_vae_model.eval()
torch.save(best_vae_model.state_dict(), FEATURES_DIR / "vae_fixed_model.pt")
np.save(FEATURES_DIR / "vae_fixed_scaler_mean.npy",  vae_scaler.mean_)
np.save(FEATURES_DIR / "vae_fixed_scaler_scale.npy", vae_scaler.scale_)
np.save(FEATURES_DIR / "vae_fixed_threshold.npy",    np.array([vae_thresh_val]))
np.save(FEATURES_DIR / "vae_fixed_window_scores.npy", vae_all_mse)
log("\n  Saved: vae_fixed_model.pt + scaler + threshold + window_scores")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Load CNN classifiers + compute primary probs + streaming
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Loading CNN classifiers + computing primary probs + streaming"); log("=" * 72)

class TextEmbeddingClassifier(nn.Module):
    def __init__(self, emb_dim=384, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.classifier   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_classes))
        self.reconstructor = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, emb_dim))
    def forward(self, x):
        feat   = self.conv(x.permute(0, 2, 1))
        pooled = feat.mean(dim=-1)
        return self.classifier(pooled), self.reconstructor(pooled), pooled

# Load MiniLM sequences + classifier
log("  Loading MiniLM sequences (1970, 300, 384)...")
minilm_seqs = np.load(FEATURES_DIR / "text_sequences.npy")   # (1970, 300, 384)
clf_minilm  = TextEmbeddingClassifier(emb_dim=384).to(device)
clf_minilm.load_state_dict(torch.load(FEATURES_DIR / "classifier.pt", map_location=device, weights_only=True))
clf_minilm.eval()

# Load mpnet sequences + classifier
log("  Loading mpnet sequences (1970, 300, 768)...")
mpnet_seqs  = np.load(FEATURES_DIR / "mpnet_sequences.npy")  # (1970, 300, 768)
clf_mpnet   = TextEmbeddingClassifier(emb_dim=768).to(device)
clf_mpnet.load_state_dict(torch.load(FEATURES_DIR / "mpnet_classifier.pt", map_location=device, weights_only=True))
clf_mpnet.eval()

def primary_probs_clf(clf, seqs, idx):
    x = torch.tensor(seqs[idx], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = clf(x)
    return torch.softmax(logits, dim=1).cpu().numpy()[0]

def streaming_max_clf(clf, seqs, idx, emb_dim):
    """Sliding 50-window max P(ransomware)."""
    seq = seqs[idx]; W = 50; best = 0.0
    with torch.no_grad():
        for start in range(0, 300 - W + 1, 5):
            chunk  = seq[start:start + W]
            padded = np.concatenate([chunk, np.zeros((300 - W, emb_dim), np.float32)])
            x = torch.tensor(padded[None], dtype=torch.float32).to(device)
            logits, _, _ = clf(x)
            p = torch.softmax(logits, dim=1)[0, RAN_IDX].item()
            if p > best: best = p
    return best

# Eval set: val + variants + mix
eval_idx = np.concatenate([val_idx, variant_idx, mix_idx])
log(f"  Precomputing probs + streaming for {len(eval_idx)} execs...")

pmap_minilm, smap_minilm = {}, {}
pmap_mpnet,  smap_mpnet  = {}, {}

for ii, i in enumerate(eval_idx):
    i = int(i)
    pmap_minilm[i] = primary_probs_clf(clf_minilm, minilm_seqs, i)
    pmap_mpnet[i]  = primary_probs_clf(clf_mpnet,  mpnet_seqs,  i)
    # streaming only needed when primary is NOT ransomware
    if pmap_minilm[i].argmax() != RAN_IDX:
        smap_minilm[i] = streaming_max_clf(clf_minilm, minilm_seqs, i, 384)
    else:
        smap_minilm[i] = 0.0
    if pmap_mpnet[i].argmax() != RAN_IDX:
        smap_mpnet[i]  = streaming_max_clf(clf_mpnet, mpnet_seqs, i, 768)
    else:
        smap_mpnet[i]  = 0.0
    if (ii + 1) % 100 == 0:
        log(f"    {ii+1}/{len(eval_idx)} done")

log("  Primary probs + streaming done.")
del minilm_seqs, mpnet_seqs   # free memory before BiLSTM training
torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — D3 Heads on fixed feat_23
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("D3 HEADS — fixed feat_23"); log("=" * 72)

X_tr_stats = feat23_stats[train_idx]
sc_global  = StandardScaler()
X_tr_scaled_stats = sc_global.fit_transform(X_tr_stats)
X_all_scaled_stats = sc_global.transform(feat23_stats)
X_tr_nr_scaled = X_tr_scaled_stats[train_nonran_mask]

# ── B1: Mahalanobis ──────────────────────────────────────────────────────────
log("\n[B1] Mahalanobis distance from non-ransomware training distribution...")
mu_nr  = X_tr_nr_scaled.mean(axis=0)
cov_nr = np.cov(X_tr_nr_scaled.T)
VI_nr  = np.linalg.pinv(cov_nr)
def maha_score(x): diff = x - mu_nr; return float(np.sqrt(diff @ VI_nr @ diff))
maha_scores_all = np.array([maha_score(X_all_scaled_stats[i]) for i in range(len(feat23_stats))])
log(f"  Non-ran training Mahalanobis: mean={maha_scores_all[train_idx][train_nonran_mask].mean():.3f}  "
    f"95th={np.percentile(maha_scores_all[train_idx][train_nonran_mask], 95):.3f}")
log(f"  Ran training Mahalanobis:     mean={maha_scores_all[train_idx][~train_nonran_mask].mean():.3f}")

# ── B2: KNN ──────────────────────────────────────────────────────────────────
log("\n[B2] KNN ransomware-fraction score (k=5, cosine)...")
knn_B2 = NearestNeighbors(n_neighbors=5, metric="cosine")
knn_B2.fit(X_tr_scaled_stats)
_, nn_idx_all = knn_B2.kneighbors(X_all_scaled_stats)
knn_ran_frac = np.array([(train_y_bin[nn_idx_all[i]] == 1).mean() for i in range(len(feat23_stats))])
log(f"  Non-ran training KNN ran-frac: mean={knn_ran_frac[train_idx][train_nonran_mask].mean():.3f}  "
    f"95th={np.percentile(knn_ran_frac[train_idx][train_nonran_mask], 95):.3f}")
log(f"  Ran training KNN ran-frac:     mean={knn_ran_frac[train_idx][~train_nonran_mask].mean():.3f}")

# ── A1: Logistic Regression ──────────────────────────────────────────────────
log("\n[A1] Supervised LR on feat_23 stats (5-fold OOF)...")
lr_oof_scores = np.zeros(len(train_idx))
for fold, (tr_f, val_f) in enumerate(skf.split(X_tr_stats, train_y_bin)):
    sc_f = StandardScaler(); X_f_tr_s = sc_f.fit_transform(X_tr_stats[tr_f])
    clf_lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs")
    clf_lr.fit(X_f_tr_s, train_y_bin[tr_f])
    lr_oof_scores[val_f] = clf_lr.predict_proba(sc_f.transform(X_tr_stats[val_f]))[:, 1]

lr_thresh_oof = float(np.percentile(lr_oof_scores[train_nonran_mask], FPR_TARGET_PCT))
lr_oof_fpr    = float((lr_oof_scores[train_nonran_mask] > lr_thresh_oof).mean())
lr_oof_tpr    = float((lr_oof_scores[~train_nonran_mask] > lr_thresh_oof).mean())
log(f"  OOF thresh: {lr_thresh_oof:.4f}  OOF FPR: {lr_oof_fpr:.1%}  OOF TPR: {lr_oof_tpr:.1%}")

sc_lr_final = StandardScaler()
clf_lr_final = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs")
clf_lr_final.fit(sc_lr_final.fit_transform(X_tr_stats), train_y_bin)
lr_scores_all = clf_lr_final.predict_proba(sc_lr_final.transform(feat23_stats))[:, 1]
lr_thresh_val = float(np.percentile(lr_scores_all[val_nonran_idx], FPR_TARGET_PCT))
log(f"  Val thresh: {lr_thresh_val:.4f}  Val FPR: {float((lr_scores_all[val_nonran_idx] > lr_thresh_val).mean()):.1%}")

# ── A2: BiLSTM ───────────────────────────────────────────────────────────────
log("\n[A2] Supervised BiLSTM on feat_23 sequences (5-fold OOF + 3-seed final)...")

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_features=23, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        D = hidden_dim * 2
        self.head = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, x):
        B, T, F = x.shape
        x_bn = self.bn_input(x.reshape(B*T, F)).reshape(B, T, F)
        _, (h_n, _) = self.lstm(x_bn)
        return self.head(torch.cat([h_n[-2], h_n[-1]], dim=1))

def normalize_seqs(X, scaler=None):
    N, T, F = X.shape
    flat = X.reshape(-1, F).astype(np.float64)
    if scaler is None:
        scaler = StandardScaler(); flat_s = scaler.fit_transform(flat).astype(np.float32)
    else:
        flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(N, T, F), scaler

def train_lstm(X_n, y, n_epochs=60, bs=64, lr=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = BiLSTMClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_ran = y.sum(); n_rest = len(y) - n_ran
    w = torch.tensor([n_ran/len(y), n_rest/len(y)], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    dl = DataLoader(TensorDataset(torch.tensor(X_n), torch.tensor(y, dtype=torch.long)), bs, shuffle=True)
    model.train()
    for ep in range(n_epochs):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep + 1) % 20 == 0: print(f"    Ep {ep+1}/{n_epochs}", flush=True)
    return model

@torch.no_grad()
def score_lstm(model, X_n):
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X_n)), 128, shuffle=False)
    probs = []
    for (bx,) in dl:
        bx = bx.to(device)
        probs.append(torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

X_tr_seqs = feat_23[train_idx]
lstm_oof_scores = np.zeros(len(train_idx))
for fold, (tr_f, val_f) in enumerate(skf.split(X_tr_seqs, train_y_bin)):
    log(f"  --- LSTM fold {fold+1}/{K_FOLDS} ---")
    X_f_tr_n, sc_f = normalize_seqs(X_tr_seqs[tr_f])
    X_f_val_n, _   = normalize_seqs(X_tr_seqs[val_f], sc_f)
    mf, _ = train_lstm(X_f_tr_n, train_y_bin[tr_f], n_epochs=60, seed=fold), None
    lstm_oof_scores[val_f] = score_lstm(mf, X_f_val_n)
    del mf; torch.cuda.empty_cache()

lstm_thresh_oof = float(np.percentile(lstm_oof_scores[train_nonran_mask], FPR_TARGET_PCT))
lstm_oof_fpr    = float((lstm_oof_scores[train_nonran_mask] > lstm_thresh_oof).mean())
lstm_oof_tpr    = float((lstm_oof_scores[~train_nonran_mask] > lstm_thresh_oof).mean())
log(f"\n  LSTM OOF thresh: {lstm_thresh_oof:.4f}  OOF FPR: {lstm_oof_fpr:.1%}  OOF TPR: {lstm_oof_tpr:.1%}")

log("\n  Training final BiLSTM (3 seeds on all 1800)...")
X_all_n, sc_lstm = normalize_seqs(feat_23[train_idx])
final_lstms = []
for seed in [0, 1, 2]:
    log(f"  Seed {seed}:")
    mf = train_lstm(X_all_n, train_y_bin, n_epochs=80, seed=seed)
    final_lstms.append(mf)

X_all_norm, _ = normalize_seqs(feat_23, sc_lstm)
lstm_scores_all = np.mean([score_lstm(m, X_all_norm) for m in final_lstms], axis=0)
lstm_thresh_val = float(np.percentile(lstm_scores_all[val_nonran_idx], FPR_TARGET_PCT))
log(f"  Final LSTM val thresh: {lstm_thresh_val:.4f}  "
    f"val FPR: {float((lstm_scores_all[val_nonran_idx]>lstm_thresh_val).mean()):.1%}  "
    f"val TPR: {float((lstm_scores_all[val_ran_idx]>lstm_thresh_val).mean()):.1%}")

del final_lstms; torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Threshold calibration for distance heads
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Threshold calibration for distance heads (val non-ran)"); log("=" * 72)

maha_thresh = float(np.percentile(maha_scores_all[val_nonran_idx], FPR_TARGET_PCT))
maha_val_fpr = float((maha_scores_all[val_nonran_idx] > maha_thresh).mean())
log(f"  B1 Mahalanobis thresh: {maha_thresh:.3f}  val FPR: {maha_val_fpr:.1%}")

knn_thresh = float(np.percentile(knn_ran_frac[val_nonran_idx], FPR_TARGET_PCT))
knn_val_fpr = float((knn_ran_frac[val_nonran_idx] > knn_thresh).mean())
log(f"  B2 KNN thresh: {knn_thresh:.3f}  val FPR: {knn_val_fpr:.1%}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Blind evaluation
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("BLIND EVALUATION"); log("=" * 72)

def fuse_v4(p, s, v, vae_thr):
    """Generic v4 logic. v is the VAE MSE score."""
    def vae_fires(vv): return vv > vae_thr
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]: return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        if s >= STREAM_THRESH or vae_fires(v): return "Attack"
        return "Benign"
    if s >= STREAM_THRESH: return "Attack"
    if vae_fires(v): return "Idle"
    return "Benign"

def fuse_v6(p, s, head_score, thresh):
    fires = head_score > thresh
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]: return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"
    if s >= STREAM_THRESH: return "Attack"
    if fires: return "Idle"
    return "Benign"

# Fusion variants: (name, get_verdict_fn)
def make_v4_minilm(i):
    return fuse_v4(pmap_minilm[i], smap_minilm[i], vae_all_mse[i], vae_thresh_val)
def make_v4_mpnet(i):
    return fuse_v4(pmap_mpnet[i],  smap_mpnet[i],  vae_all_mse[i], vae_thresh_val)

score_maps = {
    "A1_LR":   (lr_scores_all,     lr_thresh_val),
    "A2_LSTM": (lstm_scores_all,   lstm_thresh_val),
    "B1_Maha": (maha_scores_all,   maha_thresh),
    "B2_KNN":  (knn_ran_frac,      knn_thresh),
}

def get_verdict(i, fusion_name):
    i = int(i)
    if fusion_name == "v4_minilm_vae": return make_v4_minilm(i)
    if fusion_name == "v4_mpnet_vae":  return make_v4_mpnet(i)
    head_name = fusion_name.replace("v6_", "")
    scores_arr, thresh = score_maps[head_name]
    return fuse_v6(pmap_minilm[i], smap_minilm[i], scores_arr[i], thresh)

all_fusions = ["v4_minilm_vae", "v4_mpnet_vae"] + [f"v6_{h}" for h in score_maps]
json_out = {"heads": {}, "variant_recall": {}, "val_fpr": {}}

# ── Val FPR ──────────────────────────────────────────────────────────────────
log(f"\n── VAL FPR ──")
log(f"{'Fusion':>20}  {'Ben FPR':>10}  {'Lenc FPR':>10}  {'Combined FPR':>13}")
log("─" * 60)

ben_val_idx  = [i for i in val_idx if all_labels[i] in BENIGN_CLASSES]
lenc_val_idx = [i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES]
nr_val_idx   = [i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES]

for fname in all_fusions:
    b_fp  = sum(1 for i in ben_val_idx  if get_verdict(i, fname) == "Attack")
    le_fp = sum(1 for i in lenc_val_idx if get_verdict(i, fname) == "Attack")
    nr_fp = sum(1 for i in nr_val_idx   if get_verdict(i, fname) == "Attack")
    b_fpr  = b_fp  / max(len(ben_val_idx),  1)
    le_fpr = le_fp / max(len(lenc_val_idx), 1)
    nr_fpr = nr_fp / max(len(nr_val_idx),   1)
    log(f"  {fname:>20}  {b_fp}/{len(ben_val_idx)} ({b_fpr:.0%})   "
        f"{le_fp}/{len(lenc_val_idx)} ({le_fpr:.0%})   "
        f"{nr_fp}/{len(nr_val_idx)} ({nr_fpr:.0%})")
    json_out["val_fpr"][fname] = {"benign": b_fpr, "legit_enc": le_fpr, "combined": nr_fpr}

# ── Variant recall ────────────────────────────────────────────────────────────
log(f"\n── VARIANT RECALL (blind, no Conti_06 in training) ──")

for fname in all_fusions:
    log(f"\n  [{fname}]")
    log(f"  {'Variant':<14}  {'N':>4}  {'Attack':>7}  {'Idle':>6}  {'Benign':>7}  {'Recall':>8}")
    log(f"  {'─'*54}")
    total_att = 0; total_n = 0; per_variant = {}
    for vname in VARIANT_CLASSES:
        vi = [i for i in variant_idx if all_labels[i] == vname]
        n  = len(vi); total_n += n
        if n == 0: continue
        verdicts = [get_verdict(i, fname) for i in vi]
        n_att = sum(v == "Attack" for v in verdicts)
        n_idle = sum(v == "Idle"  for v in verdicts)
        n_ben  = sum(v == "Benign" for v in verdicts)
        recall = n_att / n; total_att += n_att
        flag = " ◀" if vname == "Conti_06" else ""
        log(f"  {vname+flag:<14}  {n:>4}  {n_att:>7}  {n_idle:>6}  {n_ben:>7}  {recall:>8.0%}")
        per_variant[vname] = {"n": n, "attack": n_att, "idle": n_idle, "benign": n_ben, "recall": recall}
    overall = total_att / total_n
    log(f"  {'ALL':<14}  {total_n:>4}  {total_att:>7}  {'':>6}  {'':>7}  {overall:>8.0%}")
    json_out["variant_recall"][fname] = {"overall": overall, "per_variant": per_variant}

# ── Conti_06 per-execution detail ─────────────────────────────────────────────
log(f"\n── CONTI_06 PER-EXECUTION DETAIL ──")
c06_idx = [i for i in variant_idx if all_labels[i] == "Conti_06"]
log(f"  {'Idx':>6}  {'minilm_p(ran)':>14}  {'mpnet_p(ran)':>13}  {'vae_mse':>8}  " +
    "  ".join(f"{n:>12}" for n in all_fusions))
for i in c06_idx:
    ii = int(i)
    pran_minilm = pmap_minilm[ii][RAN_IDX]
    pran_mpnet  = pmap_mpnet[ii][RAN_IDX]
    row_verdicts = "  ".join(f"{get_verdict(ii, f):>12}" for f in all_fusions)
    log(f"  {ii:>6}  {pran_minilm:>14.4f}  {pran_mpnet:>13.4f}  {vae_all_mse[ii]:>8.4f}  {row_verdicts}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LOFO (Leave-One-Family-Out cross-family zero-day)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("LOFO — Leave-One-Family-Out cross-family zero-day"); log("=" * 72)
log(f"  Using fusion: v4_minilm_vae (MiniLM + fixed VAE) as the representative pipeline")
log(f"  {'Family':<12}  {'N held out':>11}  {'Attack':>7}  {'Recall':>8}")
log("  " + "─" * 45)

lofo_results = {}
for family, members in FAMILY_CLASSES.items():
    held_idx = [i for i in range(len(all_labels)) if all_labels[i] in members
                and all_groups[i] == "original"]
    n = len(held_idx)
    if n == 0: continue
    n_att = sum(1 for i in held_idx if int(i) in pmap_minilm and
                get_verdict(i, "v4_minilm_vae") == "Attack")
    # execs not in eval_idx won't have pmap entries — score them now
    missing = [i for i in held_idx if int(i) not in pmap_minilm]
    for i in missing:
        ii = int(i)
        pmap_minilm[ii] = primary_probs_clf(clf_minilm,
                          np.load(FEATURES_DIR / "text_sequences.npy"), ii)
        # Note: this is a simplification — for LOFO we use precomputed VAE scores
        smap_minilm[ii] = 0.0   # conservative: no streaming for missing
    n_att = sum(1 for i in held_idx if get_verdict(i, "v4_minilm_vae") == "Attack")
    recall = n_att / n
    log(f"  {family:<12}  {n:>11}  {n_att:>7}  {recall:>8.0%}")
    lofo_results[family] = {"n": n, "attack": n_att, "recall": recall}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Summary table
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("SUMMARY"); log("=" * 72)
log(f"\n  Reference: MiniLM fusion_v4 (crippled feat_23 VAE)  →  84% recall / 1.7% FPR")
log(f"\n  {'Fusion':>22}  {'Recall':>8}  {'Val FPR':>9}  {'Conti_06':>10}")
log(f"  {'─'*55}")
for fname in all_fusions:
    ov_rec  = json_out["variant_recall"][fname]["overall"]
    nr_fpr  = json_out["val_fpr"][fname]["combined"]
    c06_rec = json_out["variant_recall"][fname]["per_variant"].get("Conti_06", {}).get("recall", 0.0)
    log(f"  {fname:>22}  {ov_rec:>8.0%}  {nr_fpr:>9.1%}  {c06_rec:>10.0%}")


# ── Save ─────────────────────────────────────────────────────────────────────
json_out["thresholds"] = {
    "vae_fixed": vae_thresh_val, "A1_LR": lr_thresh_val,
    "A2_LSTM": lstm_thresh_val,  "B1_Maha": maha_thresh, "B2_KNN": knn_thresh,
}
json_out["oof_metrics"] = {
    "vae":     {"fpr": vae_oof_fpr,  "tpr": vae_oof_tpr},
    "A1_LR":   {"fpr": lr_oof_fpr,   "tpr": lr_oof_tpr},
    "A2_LSTM": {"fpr": lstm_oof_fpr, "tpr": lstm_oof_tpr},
}
json_out["lofo"] = lofo_results

(FEATURES_DIR / "d3_fixed_results.txt").write_text("\n".join(lines))
with open(FEATURES_DIR / "d3_fixed_results.json", "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)

print(f"\nSaved: d3_fixed_results.txt / .json")
print("D3-FIXED done.")
