"""
ts_tcc_ranonly.py — TS-TCC Ransomware-Only Head + Two-Sided Combination
================================================================================================
Per HANDOFF item 56, TASKS B1 + B2 + C (August 10, 2026).

The VAE likelihood-ratio combination (item 55 Task C: 94%/70%) motivates
trying the same two-sided idea on TS-TCC. TS-TCC separates into an
ENCODER (self-supervised representation) and a DISTANCE HEAD (fit on top).
This script varies both pieces:

  Encoder pool (what the self-supervised pretraining sees, unlabeled):
    B1 = ALL known classes (exactly TS_TCC_FUSION's setup — re-pretrained
         here since no checkpoint was saved from that job).
    B2 = RANSOMWARE ONLY (900 training executions) — a new encoder that
         never sees non-ransomware sequences during pretraining.

  Within EACH encoder's representation space, fit TWO Mahalanobis heads:
    - dist_normal      — fit on non-ransomware training representations.
    - dist_ransomware   — fit on ransomware training representations.
  These give two results per space:
    B-variant (ransomware head alone): fires on LOW dist_ransomware,
      threshold = 5th pct of validation non-ransomware dist_ransomware.
    C-variant (two-sided): score = dist_normal - dist_ransomware, fires on
      HIGH score, threshold = 95th pct of validation non-ransomware score.

  *** CRITICAL: dist_normal and dist_ransomware are NEVER mixed across
  encoder spaces — B1's two heads only combine with each other, B2's two
  heads only combine with each other. Mahalanobis distances from different
  encoders live in different geometries at different scales. ***

3 seeds per encoder pool (6 pretrainings total). Same fusion as every other
head (frozen classifier_2way.pt + streaming + this head + binary rule).

★  LABEL-INFORMED RULE (HANDOFF item 56). Building a ransomware-distance
   head — and B2's ransomware-only pretraining pool — requires knowing
   which executions are ransomware. Every result in this script is
   LABEL-INFORMED / SUPERVISED, and must be reported as such, never as
   "unsupervised TS-TCC". Report ordering: unsupervised VAE (89/30) ->
   unsupervised TS-TCC (93/50) -> supervised generative VAE-LR (94/70) ->
   supervised TS-TCC variants (this script) -> supervised discriminative
   BiLSTM (99/90).
★  REPORTING RULE. Recall on KNOWN ransomware families is train-on-test
   for the ransomware-fitted head — diagnostic only, never a headline
   result. Only Conti_01-07 (blind) and FPR count here (LOFO is a separate,
   later task per item 56 Task D, gated on these results).
"""

import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES_DIR   = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT_HI     = 95.0   # standard direction (two-sided score)
FPR_PCT_LO     = 5.0    # inverted direction (ransomware-distance head alone)
STREAM_THRESH  = 0.65
SEEDS          = [0, 1, 2]

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]

JITTER_SIGMA = 0.03
SCALE_RATIO  = 2.0
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

log("  Loading MiniLM sequences for the frozen 2-way CNN...")
text_seqs = np.load(FEATURES_DIR / "text_sequences.npy")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx     = np.where(split["mix_mask"])[0]
eval_idx    = np.concatenate([val_idx, variant_idx, mix_idx])

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]
train_y_bin = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in train_idx])
train_nonran_global = train_idx[train_y_bin == 0]   # 900
train_ran_global    = train_idx[train_y_bin == 1]   # 900

val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ran_idx    = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])
val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
c06_idx        = [int(i) for i in variant_idx if all_labels[i] == "Conti_06"]

log(f"  Training pool: {len(train_idx)} (ran={len(train_ran_global)}, non-ran={len(train_nonran_global)})")
log(f"  Val: {len(val_idx)}  Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}, Ran={len(val_ran_idx)}")
log(f"  Variants (blind): {len(variant_idx)},  Conti_06: {len(c06_idx)}")

scaler = StandardScaler()
scaler.fit(feat_23[train_idx].reshape(-1, 23))   # shared scaler for TS-TCC input, fit on full known pool

def scale_seqs(seqs):
    N, T, C = seqs.shape
    return (scaler.transform(seqs.reshape(-1, C)).astype(np.float32).reshape(N, T, C))

feat_23_scaled = scale_seqs(feat_23)

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Frozen 2-way CNN — probs + streaming
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Loading frozen classifier_2way.pt + computing probs/streaming"); log("=" * 72)

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
clf2.load_state_dict(torch.load(FEATURES_DIR / "classifier_2way.pt",
                                map_location=device, weights_only=True))
clf2.eval()

@torch.no_grad()
def probs_2way(model, seqs, i):
    x = torch.tensor(seqs[int(i)][None], dtype=torch.float32).to(device)
    logits, _, _ = model(x)
    return torch.softmax(logits, dim=1).cpu().numpy()[0]

@torch.no_grad()
def streaming_max_2way(model, seqs, i):
    seq = seqs[int(i)]; W = 50; best = 0.0
    for start in range(0, 300 - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((300 - W, seqs.shape[-1]), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = model(x)
        p = torch.softmax(logits, dim=1)[0, 1].item()
        if p > best: best = p
    return best

log(f"  Precomputing 2-way probs + streaming for {len(eval_idx)} eval execs...")
pmap2, smap2 = {}, {}
for ii, i in enumerate(eval_idx):
    i = int(i)
    pmap2[i] = probs_2way(clf2, text_seqs, i)
    smap2[i] = 0.0 if pmap2[i].argmax() == 1 else streaming_max_2way(clf2, text_seqs, i)
    if (ii + 1) % 100 == 0: log(f"    {ii+1}/{len(eval_idx)}")
log("  Done.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TS-TCC model defs + temporal-preserving augmentation
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("TS-TCC model definitions (temporal-preserving config)"); log("=" * 72)

class Encoder(nn.Module):
    def __init__(self, in_channels=23, hidden=128, dropout=0.35):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1), nn.Dropout(dropout),
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
        x = x.permute(0, 2, 1)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        return x

class ContextTransformer(nn.Module):
    def __init__(self, d_model, h=TRANS_H, nhead=TRANS_HEADS, n_layers=TRANS_LAYERS, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, h)
        self.ctx_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=nhead, dim_feedforward=4 * h,
            dropout=dropout, activation="relu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
    def forward(self, z):
        B = z.size(0)
        h = self.proj(z)
        ctx = self.ctx_token.expand(B, -1, -1)
        out = self.transformer(torch.cat([ctx, h], dim=1))
        return out[:, 0, :]

class ProjectionHead(nn.Module):
    def __init__(self, h=TRANS_H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
    def forward(self, x): return self.net(x)

class TSTCC(nn.Module):
    def __init__(self, in_channels=23, enc_hidden=128, K=16, h=TRANS_H):
        super().__init__()
        self.encoder = Encoder(in_channels, enc_hidden)
        self.ctx_tr  = ContextTransformer(enc_hidden, h)
        self.proj    = ProjectionHead(h)
        self.K = K
        self.Wk_s2w = nn.ModuleList([nn.Linear(h, enc_hidden) for _ in range(K)])
        self.Wk_w2s = nn.ModuleList([nn.Linear(h, enc_hidden) for _ in range(K)])

def jitter_np(x, sigma=JITTER_SIGMA):
    return x + np.random.randn(*x.shape).astype(np.float32) * sigma

def scaling_np(x, ratio=SCALE_RATIO, sigma=0.1):
    B, T, C = x.shape
    factor = (np.random.randn(B, 1, C).astype(np.float32) * sigma + ratio)
    return x * factor

def weak_augment(x_np):
    return jitter_np(scaling_np(x_np), sigma=JITTER_SIGMA)

def _smooth_curve(T, n_knots=4, sigma=0.15, loc=1.0):
    knot_xs = np.linspace(0, T - 1, num=n_knots + 2)
    knot_ys = np.random.normal(loc=loc, scale=sigma, size=n_knots + 2)
    return CubicSpline(knot_xs, knot_ys)(np.arange(T))

def magnitude_warp_np(x_np, sigma=0.15, n_knots=4):
    B, T, C = x_np.shape
    out = np.empty_like(x_np)
    for i in range(B):
        curve = _smooth_curve(T, n_knots, sigma, loc=1.0)[:, None]
        out[i] = x_np[i] * curve
    return out

def time_warp_np(x_np, sigma=0.05, n_knots=4):
    B, T, C = x_np.shape
    out = np.empty_like(x_np)
    orig_steps = np.arange(T)
    for i in range(B):
        curve = np.clip(_smooth_curve(T, n_knots, sigma, loc=1.0), 1e-3, None)
        cum = np.cumsum(curve)
        cum = (cum - cum.min()) / (cum.max() - cum.min() + 1e-8) * (T - 1)
        for c in range(C):
            out[i, :, c] = np.interp(orig_steps, cum, x_np[i, :, c])
    return out

def strong_augment_temporal_preserving(x_np):
    x = magnitude_warp_np(x_np, sigma=0.15)
    x = time_warp_np(x, sigma=0.05)
    return jitter_np(x, sigma=JITTER_SIGMA)

def temporal_contrasting_loss(model, z_s, z_w, K):
    Bsz, Tp, d = z_s.shape
    t_min = max(1, int(Tp * 0.2))
    t_max = Tp - K - 1
    t = np.random.randint(t_min, t_max) if t_max > t_min else max(1, Tp - K - 1)
    c_s = model.ctx_tr(z_s[:, :t + 1, :])
    c_w = model.ctx_tr(z_w[:, :t + 1, :])
    def infonce(context, Wk_list, target_seq):
        loss = 0.0
        for k in range(K):
            pred = Wk_list[k](context)
            tgt  = target_seq[:, t + 1 + k, :]
            logits = pred @ tgt.T
            labels = torch.arange(Bsz, device=pred.device)
            loss = loss + F.cross_entropy(logits, labels)
        return loss / K
    loss_s = infonce(c_s, model.Wk_s2w, z_w)
    loss_w = infonce(c_w, model.Wk_w2s, z_s)
    return loss_s, loss_w, c_s, c_w

def contextual_contrasting_loss(proj_s, proj_w, tau=TAU):
    z = torch.cat([proj_s, proj_w], dim=0)
    z = F.normalize(z, dim=1)
    N = proj_s.size(0)
    sim = (z @ z.T) / tau
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    pos_idx = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, pos_idx)

def pretrain_tstcc(X_scaled, seed, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
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
            if len(idx) < 4: continue
            x_batch = X_scaled[idx]
            x_w = weak_augment(x_batch)
            x_s = strong_augment_temporal_preserving(x_batch)
            x_w_t = torch.tensor(x_w, dtype=torch.float32).to(device)
            x_s_t = torch.tensor(x_s, dtype=torch.float32).to(device)
            z_w = model.encoder(x_w_t).permute(0, 2, 1)
            z_s = model.encoder(x_s_t).permute(0, 2, 1)
            loss_s, loss_w, c_s, c_w = temporal_contrasting_loss(model, z_s, z_w, model.K)
            proj_s, proj_w = model.proj(c_s), model.proj(c_w)
            loss_cc = contextual_contrasting_loss(proj_s, proj_w)
            loss = LAMBDA1 * (loss_s + loss_w) + LAMBDA2 * loss_cc
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        if (ep + 1) % 20 == 0:
            log(f"      [{tag}] epoch {ep+1}/{n_epochs}  loss={ep_loss/n_batches:.4f}")
    return model

@torch.no_grad()
def extract_representations(model, X_scaled, batch_size=256):
    model.eval()
    reps = []
    for i in range(0, len(X_scaled), batch_size):
        xb = torch.tensor(X_scaled[i:i + batch_size], dtype=torch.float32).to(device)
        pooled = model.encoder(xb).mean(dim=-1)
        reps.append(pooled.cpu().numpy())
    return np.concatenate(reps, axis=0)

def fit_mahalanobis(reps_ref):
    mu = reps_ref.mean(axis=0)
    VI = np.linalg.pinv(np.cov(reps_ref.T))
    return mu, VI

def score_mahalanobis(reps, mu, VI):
    diff = reps - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, VI, diff))


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Fusion verdicts (standard + inverted)
# ════════════════════════════════════════════════════════════════════════════════

def verdict_standard(i, thr, scores, pmap, smap):
    """Fires on HIGH score (used for the two-sided C-variant)."""
    i = int(i)
    p = pmap[i]; s = smap.get(i, 0.0)
    fires = scores[i] > thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"

def verdict_inverted(i, thr, scores, pmap, smap):
    """Fires on LOW score (used for the B-variant, ransomware head alone)."""
    i = int(i)
    p = pmap[i]; s = smap.get(i, 0.0)
    fires = scores[i] < thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"

def eval_metrics(verdict_fn, thr, scores):
    verdicts_var = [verdict_fn(i, thr, scores, pmap2, smap2) for i in variant_idx]
    verdicts_c06 = [verdict_fn(i, thr, scores, pmap2, smap2) for i in c06_idx]
    verdicts_ben = [verdict_fn(i, thr, scores, pmap2, smap2) for i in val_ben_idx]
    verdicts_le  = [verdict_fn(i, thr, scores, pmap2, smap2) for i in val_lenc_idx]
    recall     = sum(v == "Attack" for v in verdicts_var) / len(variant_idx)
    c06_recall = sum(v == "Attack" for v in verdicts_c06) / len(c06_idx)
    fpr_ben    = sum(v == "Attack" for v in verdicts_ben) / len(val_ben_idx)
    fpr_lenc   = sum(v == "Attack" for v in verdicts_le) / len(val_lenc_idx)
    fpr_nr     = (sum(v == "Attack" for v in verdicts_ben) + sum(v == "Attack" for v in verdicts_le)) / len(val_nonran_idx)
    return recall, c06_recall, fpr_ben, fpr_lenc, fpr_nr


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN RUN: B1 (all-classes pool) vs B2 (ransomware-only pool), 3 seeds each
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("MAIN RUN — TS-TCC ransomware-distance head + two-sided, per encoder pool"); log("=" * 72)

ENCODER_POOLS = {
    "B1_all_classes":    feat_23_scaled[train_idx],
    "B2_ransomware_only": feat_23_scaled[train_ran_global],
}

all_results = {name: {"B_ransomware_head": [], "C_two_sided": []} for name in ENCODER_POOLS}

for pool_name, X_pool in ENCODER_POOLS.items():
    log(f"\n  ==== ENCODER POOL: {pool_name} ({X_pool.shape[0]} execs) ====")
    for seed in SEEDS:
        t0 = time.time()
        model = pretrain_tstcc(X_pool, seed, tag=f"{pool_name} seed={seed}")
        reps_all = extract_representations(model, feat_23_scaled)

        reps_normal_ref     = reps_all[train_nonran_global]
        reps_ransomware_ref = reps_all[train_ran_global]
        mu_n, VI_n = fit_mahalanobis(reps_normal_ref)
        mu_r, VI_r = fit_mahalanobis(reps_ransomware_ref)

        dist_normal     = score_mahalanobis(reps_all, mu_n, VI_n)
        dist_ransomware = score_mahalanobis(reps_all, mu_r, VI_r)
        two_sided = dist_normal - dist_ransomware

        # B-variant: ransomware head alone, inverted (fires on LOW dist_ransomware)
        thr_b = float(np.percentile(dist_ransomware[val_nonran_idx], FPR_PCT_LO))
        rec_b, c06_b, fb_b, fl_b, fn_b = eval_metrics(verdict_inverted, thr_b, dist_ransomware)
        diag_knownran_b = sum(1 for i in val_ran_idx if verdict_inverted(i, thr_b, dist_ransomware, pmap2, smap2) == "Attack") / len(val_ran_idx)

        # C-variant: two-sided, standard (fires on HIGH dist_normal - dist_ransomware)
        thr_c = float(np.percentile(two_sided[val_nonran_idx], FPR_PCT_HI))
        rec_c, c06_c, fb_c, fl_c, fn_c = eval_metrics(verdict_standard, thr_c, two_sided)

        all_results[pool_name]["B_ransomware_head"].append({
            "seed": seed, "thr": thr_b, "recall": rec_b, "c06_recall": c06_b,
            "fpr_ben": fb_b, "fpr_lenc": fl_b, "fpr_nr": fn_b,
            "diagnostic_ONLY_known_ransomware_recall": diag_knownran_b,
        })
        all_results[pool_name]["C_two_sided"].append({
            "seed": seed, "thr": thr_c, "recall": rec_c, "c06_recall": c06_c,
            "fpr_ben": fb_c, "fpr_lenc": fl_c, "fpr_nr": fn_c,
        })

        log(f"    seed={seed}  [{pool_name}]")
        log(f"      B (ransomware head, inverted): thr={thr_b:.4f}  recall={rec_b:.0%}  C06={c06_b:.0%}  "
            f"FPR_Ben={fb_b:.1%}  FPR_LEnc={fl_b:.1%}  FPR_NR={fn_b:.1%}  [diag known-ran recall={diag_knownran_b:.0%}]")
        log(f"      C (two-sided, standard):       thr={thr_c:.4f}  recall={rec_c:.0%}  C06={c06_c:.0%}  "
            f"FPR_Ben={fb_c:.1%}  FPR_LEnc={fl_c:.1%}  FPR_NR={fn_c:.1%}")
        log(f"      ({time.time()-t0:.0f}s)")

        del model; torch.cuda.empty_cache()

def summarize(metric_list, key):
    vals = np.array([m[key] for m in metric_list])
    return float(vals.mean()), float(vals.std())

log(f"\n  {'='*72}")
log("  MAIN RUN SUMMARY (mean ± std over 3 seeds) — ALL LABEL-INFORMED / SUPERVISED")
log(f"  {'='*72}")
for pool_name in ENCODER_POOLS:
    for variant in ["B_ransomware_head", "C_two_sided"]:
        rec_m, rec_s = summarize(all_results[pool_name][variant], "recall")
        c06_m, c06_s = summarize(all_results[pool_name][variant], "c06_recall")
        fn_m,  fn_s  = summarize(all_results[pool_name][variant], "fpr_nr")
        log(f"  [{pool_name:>20} / {variant:>17}]  Recall={rec_m:.0%}±{rec_s:.0%}  "
            f"C06={c06_m:.0%}±{c06_s:.0%}  FPR_NR={fn_m:.1%}±{fn_s:.1%}")

log(f"\n  Reference (all at ~5% FPR):")
log(f"    Unsupervised  VAE (all-classes/non-ran-only) = 89% / 30% C06")
log(f"    Unsupervised  TS-TCC (temporal-preserving)    = 93%±2% / 50%±14% C06")
log(f"    Supervised    VAE likelihood ratio             = 94% / 70% C06 (item 55 Task C)")
log(f"    Supervised    BiLSTM                            = 99% / 90% C06")

json_out["main_run"] = all_results

log()
log("=" * 72)
log("Task D (LOFO for the best two-sided variant) is NOT run here -- gated on")
log("these results per HANDOFF item 56. Re-submit a follow-up job once the best")
log("variant (B1/B2/C1/C2) is picked.")
log("=" * 72)

out_txt  = FEATURES_DIR / "ts_tcc_ranonly_results.txt"
out_json = FEATURES_DIR / "ts_tcc_ranonly_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("ts_tcc_ranonly.py done.")
