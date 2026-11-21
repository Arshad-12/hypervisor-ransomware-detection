"""
ts_tcc_detection.py — TS-TCC Self-Supervised Detector vs the Flat VAE
========================================================================
Per HANDOFF item 53 (professor-directed, July 18, 2026).

Implements TS-TCC (Eldele et al., "Time-Series Representation Learning via
Temporal and Contextual Contrasting", IJCAI-21, arXiv:2106.14112) as a modern
unsupervised alternative to the flat VAE anomaly head. Same input features
(feat_23, 300x23) as the VAE, for a fair comparison.

Pipeline:
  1. Pretrain TS-TCC self-supervised on TRAINING POOL ONLY (known classes,
     no labels) — weak (jitter+scale) + strong (permutation+jitter)
     augmentations, temporal contrasting (Transformer autoregressive model,
     K-step-ahead InfoNCE cross-view prediction) + contextual contrasting
     (NT-Xent on projected context vectors). 3 seeds for uncertainty.
  2. Extract representations: frozen encoder, global-average-pooled output
     (matches the paper's linear-evaluation convention — encoder features
     only, not routed back through the TC Transformer).
  3. Unsupervised detection on the representations: distance-to-normal score
     fit on TRAINING NON-RANSOMWARE representations only. Both Mahalanobis
     and KNN(k=5, cosine) are computed (same two distance heads D3-FIXED
     used for feat_23, B1/B2) so the comparison isn't tied to one metric's
     quirks.
  4. Threshold = 95th pct of VALIDATION non-ransomware scores (identical
     discipline to the VAE/BiLSTM runs — calibrate on val, test blind).
  5. Blind eval: variant recall, Conti_06 per-exec, Benign/LEnc/combined
     FPR, mean ± std over 3 seeds. 6-family LOFO (single seed per fold —
     documented compute trade-off, see NOTE below).
  6. Head-to-head vs VAE (89% / 30% C06 / 5.0% FPR) and BiLSTM-binary
     (99% / 90% C06 / 5.0% FPR).

Implementation choices / simplifications vs. the paper (documented for
honesty, none of them touch the ★-clean discipline):
  - Encoder + Transformer weights are SHARED between the weak and strong
    branches (standard siamese-style contrastive setup; the paper's Figure 1
    shows two symmetric boxes but does not mandate separate weights).
  - The anchor timestep t for the K-step-ahead prediction is sampled
    uniformly at random per batch (paper does not specify the sampling rule).
  - LOFO folds use a single seed (seed 0), not 3 — full 3-seed LOFO would be
    3x18 = correspondingly more full self-supervised pretrainings; flag to
    Arshad if he wants the full uncertainty estimate on LOFO too.
  - Hyperparameters follow the paper's HAR/Epilepsy configuration (closest
    sequence length to our 300 timesteps): permutation M=10, weak scaling
    ratio=2, Transformer hidden h=100, L=4 layers, 4 heads, K=40% of the
    encoder's temporal output length, lambda1=1, lambda2=0.7, tau=0.2,
    Adam lr=3e-4 wd=3e-4 betas=(0.9,0.99), batch=128, 40 epochs.

★  CORE RULE: pretrain on known classes only; Conti_06 NEVER in training or
   threshold calibration; threshold from validation non-ransomware only;
   blind eval on ALL variants + LOFO; FPR always reported.
"""

import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT      = 95.0
SEEDS        = [0, 1, 2]

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]
LOFO_FAMILIES        = {
    "WannaCry": ["WannaCry"], "Ryuk": ["Ryuk"], "REvil": ["REvil"],
    "LockBit": ["LockBit"], "Darkside": ["Darkside"], "Conti": ["Conti"],
}

# TS-TCC hyperparameters (paper, HAR/Epilepsy configuration)
PERM_M       = 10
SCALE_RATIO  = 2.0
JITTER_SIGMA = 0.03
TRANS_H      = 100
TRANS_HEADS  = 4
TRANS_LAYERS = 4
LAMBDA1      = 1.0
LAMBDA2      = 0.7
TAU          = 0.2
N_EPOCHS     = 40
BATCH_SIZE   = 128
LR           = 3e-4
WD           = 3e-4

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
# SECTION 1 — Load data
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72); log("Loading data"); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
meta_df    = pd.read_csv(FEATURES_DIR / "metadata.csv")

log("  Loading feat_23 (1970, 300, 23)...")
t0 = time.time()
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])
log(f"  Loaded in {time.time()-t0:.1f}s")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx     = np.where(split["mix_mask"])[0]

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]
train_y_bin = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in train_idx])
train_nonran_mask = (train_y_bin == 0)

val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ran_idx    = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])
val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
c06_idx        = [int(i) for i in variant_idx if all_labels[i] == "Conti_06"]

log(f"  Training pool: {len(train_idx)} (ran={train_y_bin.sum()}, non-ran={train_nonran_mask.sum()})")
log(f"  Val: {len(val_idx)}  Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}, Ran={len(val_ran_idx)}")
log(f"  Variants: {len(variant_idx)},  Conti_06: {len(c06_idx)}")

# StandardScaler fit on training pool only — same discipline as the VAE
scaler = StandardScaler()
scaler.fit(feat_23[train_idx].reshape(-1, 23))

def scale_seqs(seqs):
    N, T, C = seqs.shape
    return (scaler.transform(seqs.reshape(-1, C)).astype(np.float32)
            .reshape(N, T, C))

feat_23_scaled = scale_seqs(feat_23)

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model definitions
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Model definitions"); log("=" * 72)

class Encoder(nn.Module):
    """3-block 1D-CNN, Wang et al. 2017 FCN-style (as used in TS-TCC)."""
    def __init__(self, in_channels=23, hidden=128, dropout=0.35):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, hidden, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        return x  # (B, hidden, T')


class ContextTransformer(nn.Module):
    """Autoregressive Transformer (Fig. 2 of the paper): projects encoder
    features to hidden dim h, prepends a learned context token, runs L
    pre-norm Transformer layers, returns the context token's final state."""
    def __init__(self, d_model, h=TRANS_H, nhead=TRANS_HEADS, n_layers=TRANS_LAYERS, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, h)
        self.ctx_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=nhead, dim_feedforward=4 * h,
            dropout=dropout, activation="relu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.h = h
    def forward(self, z):
        # z: (B, T, d_model) — features up to some timestep (already sliced by caller)
        B = z.size(0)
        h = self.proj(z)
        ctx = self.ctx_token.expand(B, -1, -1)
        seq = torch.cat([ctx, h], dim=1)
        out = self.transformer(seq)
        return out[:, 0, :]  # (B, h) context vector c_t


class ProjectionHead(nn.Module):
    def __init__(self, h=TRANS_H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
    def forward(self, x): return self.net(x)


class TSTCC(nn.Module):
    """Bundles encoder + shared context Transformer + K future-step
    predictors (two directions) + contextual-contrasting projection head."""
    def __init__(self, in_channels=23, enc_hidden=128, K=16, h=TRANS_H):
        super().__init__()
        self.encoder = Encoder(in_channels, enc_hidden)
        self.ctx_tr  = ContextTransformer(enc_hidden, h)
        self.proj    = ProjectionHead(h)
        self.K = K
        # future-step predictors: strong-context -> weak target, and vice versa
        self.Wk_s2w = nn.ModuleList([nn.Linear(h, enc_hidden) for _ in range(K)])
        self.Wk_w2s = nn.ModuleList([nn.Linear(h, enc_hidden) for _ in range(K)])


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Augmentations
# ════════════════════════════════════════════════════════════════════════════════
log(); log("Augmentations: weak = jitter+scale, strong = permutation+jitter")

def jitter_np(x, sigma=JITTER_SIGMA):
    return x + np.random.randn(*x.shape).astype(np.float32) * sigma

def scaling_np(x, ratio=SCALE_RATIO, sigma=0.1):
    # x: (B, T, C) — one random factor per sample per channel
    B, T, C = x.shape
    factor = (np.random.randn(B, 1, C).astype(np.float32) * sigma + ratio)
    return x * factor

def weak_augment(x_np):
    return jitter_np(scaling_np(x_np), sigma=JITTER_SIGMA)

def permutation_np(x_np, max_segments=PERM_M):
    B, T, C = x_np.shape
    out = np.empty_like(x_np)
    for i in range(B):
        n_seg = np.random.randint(1, max_segments + 1)
        if n_seg == 1 or T <= n_seg:
            out[i] = x_np[i]; continue
        splits = np.sort(np.random.choice(np.arange(1, T), n_seg - 1, replace=False))
        segs = np.split(x_np[i], splits, axis=0)
        order = np.random.permutation(len(segs))
        out[i] = np.concatenate([segs[j] for j in order], axis=0)
    return out

def strong_augment(x_np):
    return jitter_np(permutation_np(x_np), sigma=JITTER_SIGMA)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Losses
# ════════════════════════════════════════════════════════════════════════════════

def temporal_contrasting_loss(model, z_s, z_w, K):
    """z_s, z_w: (B, T', d) full encoder feature sequences for strong/weak
    views. Randomly sample anchor t, predict K future steps of the OTHER
    branch from THIS branch's context (eq. 1 and 2)."""
    Bsz, Tp, d = z_s.shape
    t_min = max(1, int(Tp * 0.2))
    t_max = Tp - K - 1
    if t_max <= t_min:
        t = max(1, Tp - K - 1)
    else:
        t = np.random.randint(t_min, t_max)

    c_s = model.ctx_tr(z_s[:, :t + 1, :])   # context from strong branch up to t
    c_w = model.ctx_tr(z_w[:, :t + 1, :])   # context from weak branch up to t

    def infonce(context, Wk_list, target_seq):
        loss = 0.0
        for k in range(K):
            pred = Wk_list[k](context)               # (B, d)
            tgt  = target_seq[:, t + 1 + k, :]        # (B, d)
            logits = pred @ tgt.T                     # (B, B) — batch negatives
            labels = torch.arange(Bsz, device=pred.device)
            loss = loss + F.cross_entropy(logits, labels)
        return loss / K

    loss_s = infonce(c_s, model.Wk_s2w, z_w)   # strong context predicts weak future
    loss_w = infonce(c_w, model.Wk_w2s, z_s)   # weak context predicts strong future
    return loss_s, loss_w, c_s, c_w


def contextual_contrasting_loss(proj_s, proj_w, tau=TAU):
    """NT-Xent over the 2N batch of projected contexts (eq. 5)."""
    z = torch.cat([proj_s, proj_w], dim=0)                    # (2N, h)
    z = F.normalize(z, dim=1)
    N = proj_s.size(0)
    sim = (z @ z.T) / tau                                     # (2N, 2N)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    pos_idx = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, pos_idx)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Pretraining + representation extraction
# ════════════════════════════════════════════════════════════════════════════════

def pretrain_tstcc(X_scaled, seed, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose_tag=""):
    """X_scaled: (N, 300, 23) numpy, already StandardScaler-scaled. Returns
    the trained model (encoder is what matters for downstream use)."""
    torch.manual_seed(seed); np.random.seed(seed)

    # feature length after encoder — dummy forward pass
    with torch.no_grad():
        dummy = torch.zeros(2, X_scaled.shape[1], X_scaled.shape[2])
        Tp = Encoder()(dummy).shape[-1]
    K = max(2, round(0.4 * Tp))

    model = TSTCC(in_channels=X_scaled.shape[2], K=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.99))

    N = X_scaled.shape[0]
    n_batches = max(1, N // batch_size)

    for ep in range(n_epochs):
        perm = np.random.permutation(N)
        ep_loss = 0.0
        model.train()
        for bi in range(n_batches):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            if len(idx) < 4:
                continue
            x_batch = X_scaled[idx]
            x_w = weak_augment(x_batch)
            x_s = strong_augment(x_batch)
            x_w_t = torch.tensor(x_w, dtype=torch.float32).to(device)
            x_s_t = torch.tensor(x_s, dtype=torch.float32).to(device)

            z_w = model.encoder(x_w_t).permute(0, 2, 1)  # (B, T', d)
            z_s = model.encoder(x_s_t).permute(0, 2, 1)

            loss_s, loss_w, c_s, c_w = temporal_contrasting_loss(model, z_s, z_w, model.K)
            proj_s, proj_w = model.proj(c_s), model.proj(c_w)
            loss_cc = contextual_contrasting_loss(proj_s, proj_w)

            loss = LAMBDA1 * (loss_s + loss_w) + LAMBDA2 * loss_cc

            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

        if (ep + 1) % 10 == 0:
            log(f"    [{verbose_tag} seed={seed}] epoch {ep+1}/{n_epochs}  loss={ep_loss/n_batches:.4f}")

    return model


@torch.no_grad()
def extract_representations(model, X_scaled, batch_size=256):
    """Frozen encoder, global-average-pooled — the paper's linear-eval
    representation convention."""
    model.eval()
    reps = []
    for i in range(0, len(X_scaled), batch_size):
        xb = torch.tensor(X_scaled[i:i + batch_size], dtype=torch.float32).to(device)
        z = model.encoder(xb)          # (B, hidden, T')
        pooled = z.mean(dim=-1)        # (B, hidden)
        reps.append(pooled.cpu().numpy())
    return np.concatenate(reps, axis=0)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Distance-to-normal detectors
# ════════════════════════════════════════════════════════════════════════════════

def fit_mahalanobis(reps_normal):
    mu = reps_normal.mean(axis=0)
    cov = np.cov(reps_normal.T)
    VI = np.linalg.pinv(cov)
    return mu, VI

def score_mahalanobis(reps, mu, VI):
    diff = reps - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, VI, diff))

def fit_knn(reps_normal, k=5):
    nn_model = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn_model.fit(reps_normal)
    return nn_model

def score_knn(reps, nn_model):
    dist, _ = nn_model.kneighbors(reps)
    return dist.mean(axis=1)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main run: 3 seeds, full training pool
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("MAIN RUN — 3 seeds, TS-TCC pretrained on full training pool"); log("=" * 72)

X_train_scaled = feat_23_scaled[train_idx]

seed_metrics = {"maha": [], "knn": []}
seed_c06_detail = []
seed_per_variant = {"maha": [], "knn": []}

for seed in SEEDS:
    t0 = time.time()
    log(f"\n  --- Seed {seed} ---")
    model = pretrain_tstcc(X_train_scaled, seed, verbose_tag="main")
    reps_all = extract_representations(model, feat_23_scaled)   # (1970, 128)
    reps_train_nonran = reps_all[train_idx][train_nonran_mask]

    mu, VI = fit_mahalanobis(reps_train_nonran)
    knn_model = fit_knn(reps_train_nonran, k=5)

    maha_scores = score_mahalanobis(reps_all, mu, VI)
    knn_scores  = score_knn(reps_all, knn_model)

    for name, scores in [("maha", maha_scores), ("knn", knn_scores)]:
        thr = float(np.percentile(scores[val_nonran_idx], FPR_PCT))
        var_att = sum(1 for i in variant_idx if scores[i] > thr)
        c06_att = sum(1 for i in c06_idx if scores[i] > thr)
        b_fp    = sum(1 for i in val_ben_idx if scores[i] > thr)
        le_fp   = sum(1 for i in val_lenc_idx if scores[i] > thr)
        nr_fp   = b_fp + le_fp

        recall     = var_att / len(variant_idx)
        c06_recall = c06_att / len(c06_idx)
        fpr_ben    = b_fp / len(val_ben_idx)
        fpr_lenc   = le_fp / len(val_lenc_idx)
        fpr_nr     = nr_fp / len(val_nonran_idx)

        seed_metrics[name].append({
            "thr": thr, "recall": recall, "c06_recall": c06_recall,
            "fpr_ben": fpr_ben, "fpr_lenc": fpr_lenc, "fpr_nr": fpr_nr,
        })

        per_variant = {}
        for vname in VARIANT_CLASSES:
            vi = [int(i) for i in variant_idx if all_labels[i] == vname]
            n_att = sum(1 for i in vi if scores[i] > thr)
            per_variant[vname] = {"n": len(vi), "attack": n_att, "recall": n_att / len(vi)}
        seed_per_variant[name].append(per_variant)

        log(f"    [{name}] thr={thr:.4f}  recall={recall:.0%}  C06={c06_recall:.0%}  "
            f"FPR_Ben={fpr_ben:.1%}  FPR_LEnc={fpr_lenc:.1%}  FPR_NR={fpr_nr:.1%}")

    # Conti_06 per-exec detail (using Mahalanobis, the primary head)
    maha_thr = seed_metrics["maha"][-1]["thr"]
    detail = []
    for i in c06_idx:
        detail.append({
            "idx": i, "maha": float(maha_scores[i]), "maha_fires": bool(maha_scores[i] > maha_thr),
            "knn": float(knn_scores[i]), "knn_fires": bool(knn_scores[i] > seed_metrics["knn"][-1]["thr"]),
        })
    seed_c06_detail.append(detail)

    log(f"  Seed {seed} done in {time.time()-t0:.1f}s")
    del model; torch.cuda.empty_cache()

def summarize(metric_list, key):
    vals = np.array([m[key] for m in metric_list])
    return float(vals.mean()), float(vals.std())

log(f"\n  {'='*72}")
log("  MAIN RUN SUMMARY (mean ± std over 3 seeds)")
log(f"  {'='*72}")
for name in ["maha", "knn"]:
    rec_m, rec_s = summarize(seed_metrics[name], "recall")
    c06_m, c06_s = summarize(seed_metrics[name], "c06_recall")
    fb_m,  fb_s  = summarize(seed_metrics[name], "fpr_ben")
    fl_m,  fl_s  = summarize(seed_metrics[name], "fpr_lenc")
    fn_m,  fn_s  = summarize(seed_metrics[name], "fpr_nr")
    log(f"  [{name:>5}]  Recall={rec_m:.0%}±{rec_s:.0%}  C06={c06_m:.0%}±{c06_s:.0%}  "
        f"FPR_Ben={fb_m:.1%}±{fb_s:.1%}  FPR_LEnc={fl_m:.1%}±{fl_s:.1%}  FPR_NR={fn_m:.1%}±{fn_s:.1%}")

log(f"\n  Head-to-head reference:  VAE  = 89% / 30% C06 / 5.0% FPR_NR")
log(f"                           BiLSTM = 99% / 90% C06 / 5.0% FPR_NR")

json_out["main_run"] = {
    "seed_metrics": seed_metrics,
    "seed_per_variant": seed_per_variant,
    "seed_c06_detail": seed_c06_detail,
}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — 6-Family LOFO (single seed per fold — see NOTE in docstring)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("6-FAMILY LOFO (single seed=0 per fold)"); log("=" * 72)

lofo_results = {}
for family, members in LOFO_FAMILIES.items():
    t0 = time.time()
    log(f"\n  === LOFO: hold out {family} ===")
    held_mask   = np.array([all_labels[i] in members for i in train_idx])
    held_global = train_idx[np.where(held_mask)[0]]
    fold_tr_idx = train_idx[np.where(~held_mask)[0]]

    model = pretrain_tstcc(feat_23_scaled[fold_tr_idx], seed=0, verbose_tag=f"LOFO-{family}")
    reps_all = extract_representations(model, feat_23_scaled)

    fold_train_nonran_mask = np.array([all_labels[i] not in RANSOMWARE_CLASSES for i in fold_tr_idx])
    reps_train_nonran = reps_all[fold_tr_idx][fold_train_nonran_mask]
    mu, VI = fit_mahalanobis(reps_train_nonran)
    maha_scores = score_mahalanobis(reps_all, mu, VI)

    thr = float(np.percentile(maha_scores[val_nonran_idx], FPR_PCT))
    n_att = sum(1 for i in held_global if maha_scores[i] > thr)
    n_held = len(held_global)
    recall = n_att / n_held

    b_fp  = sum(1 for i in val_ben_idx  if maha_scores[i] > thr)
    le_fp = sum(1 for i in val_lenc_idx if maha_scores[i] > thr)

    log(f"    N={n_held}  Attack={n_att}  Recall={recall:.0%}  "
        f"FPR_Ben={b_fp/len(val_ben_idx):.1%}  FPR_LEnc={le_fp/len(val_lenc_idx):.1%}  "
        f"({time.time()-t0:.0f}s)")

    lofo_results[family] = {
        "n": n_held, "attack": n_att, "recall": recall,
        "fpr_ben": b_fp / len(val_ben_idx), "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": (b_fp + le_fp) / len(val_nonran_idx),
    }
    del model; torch.cuda.empty_cache()

json_out["lofo_maha"] = lofo_results

log(f"\n  LOFO SUMMARY (Mahalanobis head, single seed):")
log(f"  {'Family':<12}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
log("  " + "─" * 54)
recalls = []
for fam, r in lofo_results.items():
    log(f"  {fam:<12}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  "
        f"{r['fpr_lenc']:>10.1%}  {r['fpr_nr']:>8.1%}")
    recalls.append(r["recall"])
log(f"  {'MEAN':<12}  {np.mean(recalls):>8.0%}")
log(f"\n  Reference — BiLSTM-binary LOFO: 100% all families. VAE LOFO mean: 70%.")


# ════════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════════
out_txt  = FEATURES_DIR / "ts_tcc_detection_results.txt"
out_json = FEATURES_DIR / "ts_tcc_detection_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("ts_tcc_detection.py done.")
