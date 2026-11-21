"""
heldout_fpr_split.py — Non-Circular FPR: Calibrate on Half, Measure on the Other Half
================================================================================================
Per HANDOFF item 64 (professor comment, August 2026).

The comment: "A threshold chosen as the 95th percentile of validation errors will produce
approximately 5% FPR on that validation set BY DESIGN. Report FPR on a separate, untouched
benign and legitimate-encryption test set." She's correct — every reported FPR in this project
is measured on the same 180 validation non-ransomware executions used to set the threshold, so
5% is guaranteed by construction, not a measurement. No genuinely untouched benign set exists
(mix executions contain ransomware; held-out variants are all ransomware), so this task instead
splits the existing 180 into two halves, calibrates on one, measures on the other, and repeats
over many random splits so a single unlucky/lucky split doesn't drive the conclusion.

★ NO RETRAINING for 3 of 4 heads — reuses existing saved per-exec score arrays:
    VAE (non-ransomware-only):  vae_nonran_window_scores.npy
    TS-TCC (unsupervised):      ts_tcc_unsup_window_scores.npy   (saved by item 58's ablation)
    BiLSTM (deployed):          lstm_scores_all.npy
  ONE exception: TS-TCC (semi-supervised, full_label head) was never persisted to disk by
  `ts_tcc_semisup.py` (only summary stats were logged). This script reproduces that exact
  config once — seed=0, matching the per-seed row that hit 99%/90% (thr=0.1832) — purely to
  capture per-exec scores, and saves them this time for future reuse. Not a new experiment.

★ Streaming is NOT recomputed here. Item 63's diagnostic already proved, on this exact
  evaluation set (val Benign/LEnc/Ransomware + all 70 Conti variants), that the streaming
  channel scores EXACTLY 0.0000 for every execution with no exceptions. Recomputing it here
  would reproduce that same zero at real compute cost for no informational gain — it is fixed
  at 0.0 for all executions, cited from item 63 rather than silently assumed.

Per split (repeated N_SPLITS=20 times, stratified so each half gets ~45 Benign + ~45 LEnc):
  1. Random stratified 50/50 split of the 180 validation non-ransomware executions.
  2. Threshold = 95th pct of each head's score on half A ONLY.
  3. Measure: FPR on half B (never seen by this threshold) — Benign and LEnc separately and
     combined — plus variant recall (70 Conti_01-07) and Conti_06 recall, all at that threshold.
Report mean ± std over the 20 splits, per head, alongside the existing headline figures.

★ Scope note: this addresses the calibration/measurement CIRCULARITY the professor flagged.
  It does NOT address the separate session-exposure finding (item 59) — both halves still
  contain executions with same-session siblings in the training pool. Two distinct problems.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT   = 95.0
N_SPLITS  = 20
TSTCC_SEED = 0

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES

# Existing headline figures (main pipeline, threshold on the FULL 180-exec val non-ran set)
HEADLINE = {
    "VAE_nonran_only":  {"recall": 0.89, "c06": 0.30, "fpr": 0.050},
    "TS_TCC_unsup":      {"recall": 0.94, "c06": 0.60, "fpr": 0.050},  # seed-0 reproduction (matches this script's reused array)
    "TS_TCC_semisup":    {"recall": 0.99, "c06": 0.90, "fpr": 0.050},  # seed-0, full_label
    "BiLSTM":            {"recall": 0.99, "c06": 0.90, "fpr": 0.050},
}

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
# SECTION 1 — Load data + classifier (for the classifier-alone escalation path)
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72); log("Loading data + existing artifacts"); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
meta_df    = pd.read_csv(FEATURES_DIR / "metadata.csv")
text_seqs  = np.load(FEATURES_DIR / "text_sequences.npy")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
train_idx   = split["train_idx"]
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]

val_ben_idx  = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
conti06_idx  = np.array([i for i in variant_idx if all_labels[i] == "Conti_06"])
log(f"  Val Benign={len(val_ben_idx)}  Val LEnc={len(val_lenc_idx)}  "
    f"Variants={len(variant_idx)} (Conti_06={len(conti06_idx)})")

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

@torch.no_grad()
def probs_2way(i):
    x = torch.tensor(text_seqs[int(i)][None], dtype=torch.float32).to(device)
    logits, _, _ = clf2(x)
    return torch.softmax(logits, dim=1)[0, 1].item()

eval_idx = np.concatenate([val_ben_idx, val_lenc_idx, variant_idx])
log(f"  Precomputing classifier P(ransomware) for {len(eval_idx)} execs "
    f"(streaming NOT recomputed — item 63 proved it's exactly 0.0 on this exact set)...")
pran = {int(i): probs_2way(int(i)) for i in eval_idx}
log("  Done.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Existing score arrays (VAE, TS-TCC-unsup, BiLSTM) — no retrain
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Loading existing per-exec score arrays (no retrain)"); log("=" * 72)

vae_nonran_scores = np.load(FEATURES_DIR / "vae_nonran_window_scores.npy")
tstcc_unsup_scores = np.load(FEATURES_DIR / "ts_tcc_unsup_window_scores.npy")
lstm_scores_all    = np.load(FEATURES_DIR / "lstm_scores_all.npy")
log(f"  vae_nonran_window_scores.npy   {vae_nonran_scores.shape}")
log(f"  ts_tcc_unsup_window_scores.npy {tstcc_unsup_scores.shape}")
log(f"  lstm_scores_all.npy            {lstm_scores_all.shape}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TS-TCC semi-supervised (full_label): ONE reproduction, seed=0
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("TS-TCC semi-supervised (full_label, seed=0) — single reproduction"); log("=" * 72)

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx_mask = np.where(train_mask)[0]
log(f"  Mask-derived train pool (matches ts_tcc_semisup.py's original method exactly): {len(train_idx_mask)} execs")
log(f"  split[\"train_idx\"] (loaded above, used by other scripts in this project): {len(train_idx)} execs")
if set(train_idx_mask.tolist()) != set(train_idx.tolist()):
    n_diff = len(set(train_idx_mask.tolist()) ^ set(train_idx.tolist()))
    log(f"  NOTE: these are NOT identical ({n_diff} indices differ) — using the MASK-DERIVED "
        f"version for this reproduction since that is what ts_tcc_semisup.py actually used. "
        f"This discrepancy between the two train-pool derivations used across this project's "
        f"scripts was not previously known and is flagged here for the record, not resolved.")
else:
    log(f"  Confirmed identical to split[\"train_idx\"].")
train_idx_tstcc = train_idx_mask
train_y_bin = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in train_idx_tstcc])

log("  Loading feat_23...")
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])

scaler = StandardScaler()
scaler.fit(feat_23[train_idx_tstcc].reshape(-1, 23))
def scale_seqs(seqs):
    N, T, C = seqs.shape
    return scaler.transform(seqs.reshape(-1, C)).astype(np.float32).reshape(N, T, C)
feat_23_scaled = scale_seqs(feat_23)

TRANS_H, TRANS_HEADS, TRANS_LAYERS = 100, 4, 4
LAMBDA1, LAMBDA2, TAU = 1.0, 0.7, 0.2
N_EPOCHS, BATCH_SIZE, LR, WD = 40, 128, 3e-4, 3e-4
JITTER_SIGMA, SCALE_RATIO = 0.03, 2.0

class Encoder(nn.Module):
    def __init__(self, in_channels=23, hidden=128, dropout=0.35):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2, padding=1), nn.Dropout(dropout))
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2, padding=1))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, hidden, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.MaxPool1d(2, 2, padding=1))
    def forward(self, x):
        x = x.permute(0, 2, 1); x = self.conv1(x); x = self.conv2(x); return self.conv3(x)

class ContextTransformer(nn.Module):
    def __init__(self, d_model, h=TRANS_H, nhead=TRANS_HEADS, n_layers=TRANS_LAYERS, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, h)
        self.ctx_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=h, nhead=nhead, dim_feedforward=4*h,
                                            dropout=dropout, activation="relu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
    def forward(self, z):
        B = z.size(0); h = self.proj(z); ctx = self.ctx_token.expand(B, -1, -1)
        return self.transformer(torch.cat([ctx, h], dim=1))[:, 0, :]

class ProjectionHead(nn.Module):
    def __init__(self, h=TRANS_H):
        super().__init__(); self.net = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
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

def jitter_np(x, sigma=JITTER_SIGMA): return x + np.random.randn(*x.shape).astype(np.float32) * sigma
def scaling_np(x, ratio=SCALE_RATIO, sigma=0.1):
    B, T, C = x.shape
    return x * (np.random.randn(B, 1, C).astype(np.float32) * sigma + ratio)
def weak_augment(x_np): return jitter_np(scaling_np(x_np), sigma=JITTER_SIGMA)

def _smooth_curve(T, n_knots=4, sigma=0.15, loc=1.0):
    knot_xs = np.linspace(0, T - 1, num=n_knots + 2)
    knot_ys = np.random.normal(loc=loc, scale=sigma, size=n_knots + 2)
    return CubicSpline(knot_xs, knot_ys)(np.arange(T))

def magnitude_warp_np(x_np, sigma=0.15, n_knots=4):
    B, T, C = x_np.shape; out = np.empty_like(x_np)
    for i in range(B): out[i] = x_np[i] * _smooth_curve(T, n_knots, sigma, loc=1.0)[:, None]
    return out

def time_warp_np(x_np, sigma=0.05, n_knots=4):
    B, T, C = x_np.shape; out = np.empty_like(x_np); orig_steps = np.arange(T)
    for i in range(B):
        curve = np.clip(_smooth_curve(T, n_knots, sigma, loc=1.0), 1e-3, None)
        cum = np.cumsum(curve); cum = (cum - cum.min()) / (cum.max() - cum.min() + 1e-8) * (T - 1)
        for c in range(C): out[i, :, c] = np.interp(orig_steps, cum, x_np[i, :, c])
    return out

def strong_augment_temporal_preserving(x_np):
    x = magnitude_warp_np(x_np, sigma=0.15); x = time_warp_np(x, sigma=0.05)
    return jitter_np(x, sigma=JITTER_SIGMA)

def temporal_contrasting_loss(model, z_s, z_w, K):
    Bsz, Tp, d = z_s.shape
    t_min = max(1, int(Tp * 0.2)); t_max = Tp - K - 1
    t = np.random.randint(t_min, t_max) if t_max > t_min else max(1, Tp - K - 1)
    c_s = model.ctx_tr(z_s[:, :t+1, :]); c_w = model.ctx_tr(z_w[:, :t+1, :])
    def infonce(context, Wk_list, target_seq):
        loss = 0.0
        for k in range(K):
            pred = Wk_list[k](context); tgt = target_seq[:, t+1+k, :]
            logits = pred @ tgt.T; labels = torch.arange(Bsz, device=pred.device)
            loss = loss + F.cross_entropy(logits, labels)
        return loss / K
    return infonce(c_s, model.Wk_s2w, z_w), infonce(c_w, model.Wk_w2s, z_s), c_s, c_w

def contextual_contrasting_loss(proj_s, proj_w, tau=TAU):
    z = torch.cat([proj_s, proj_w], dim=0); z = F.normalize(z, dim=1)
    N = proj_s.size(0); sim = (z @ z.T) / tau
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device); sim.masked_fill_(mask, float("-inf"))
    pos_idx = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, pos_idx)

def pretrain_tstcc(X_scaled, seed, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    torch.manual_seed(seed); np.random.seed(seed)
    with torch.no_grad():
        dummy = torch.zeros(2, X_scaled.shape[1], X_scaled.shape[2])
        Tp = Encoder()(dummy).shape[-1]
    K = max(2, round(0.4 * Tp))
    model = TSTCC(in_channels=X_scaled.shape[2], K=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.99))
    N = X_scaled.shape[0]; n_batches = max(1, N // batch_size)
    for ep in range(n_epochs):
        perm = np.random.permutation(N); ep_loss = 0.0; model.train()
        for bi in range(n_batches):
            idx = perm[bi*batch_size:(bi+1)*batch_size]
            if len(idx) < 4: continue
            x_batch = X_scaled[idx]
            x_w = torch.tensor(weak_augment(x_batch), dtype=torch.float32).to(device)
            x_s = torch.tensor(strong_augment_temporal_preserving(x_batch), dtype=torch.float32).to(device)
            z_w = model.encoder(x_w).permute(0, 2, 1); z_s = model.encoder(x_s).permute(0, 2, 1)
            loss_s, loss_w, c_s, c_w = temporal_contrasting_loss(model, z_s, z_w, model.K)
            proj_s, proj_w = model.proj(c_s), model.proj(c_w)
            loss = LAMBDA1 * (loss_s + loss_w) + LAMBDA2 * contextual_contrasting_loss(proj_s, proj_w)
            opt.zero_grad(); loss.backward(); opt.step(); ep_loss += loss.item()
        if (ep+1) % 20 == 0: log(f"      epoch {ep+1}/{n_epochs}  loss={ep_loss/n_batches:.4f}")
    return model

@torch.no_grad()
def extract_representations(model, X_scaled, batch_size=256):
    model.eval(); reps = []
    for i in range(0, len(X_scaled), batch_size):
        xb = torch.tensor(X_scaled[i:i+batch_size], dtype=torch.float32).to(device)
        reps.append(model.encoder(xb).mean(dim=-1).cpu().numpy())
    return np.concatenate(reps, axis=0)

class SupervisedHead(nn.Module):
    def __init__(self, in_dim=128, hidden=64, n_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
    def forward(self, x): return self.net(x)

def train_head(reps_train, y_train, seed, n_epochs=100, lr=1e-3, batch_size=32):
    torch.manual_seed(seed); np.random.seed(seed)
    head = SupervisedHead(in_dim=reps_train.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    n_ran = y_train.sum(); n_rest = len(y_train) - n_ran
    w = torch.tensor([n_ran/len(y_train), n_rest/len(y_train)], dtype=torch.float32).to(device)
    dl = DataLoader(TensorDataset(torch.tensor(reps_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.long)), min(batch_size, len(y_train)), shuffle=True)
    for ep in range(n_epochs):
        head.train()
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); F.cross_entropy(head(bx), by, weight=w).backward(); opt.step()
    return head

@torch.no_grad()
def head_probs_all(head, reps_all, batch_size=256):
    head.eval(); probs = []
    for i in range(0, len(reps_all), batch_size):
        xb = torch.tensor(reps_all[i:i+batch_size], dtype=torch.float32).to(device)
        probs.append(torch.softmax(head(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

model = pretrain_tstcc(feat_23_scaled[train_idx_tstcc], seed=TSTCC_SEED)
reps_all = extract_representations(model, feat_23_scaled)
head = train_head(reps_all[train_idx_tstcc], train_y_bin, seed=TSTCC_SEED)
tstcc_semisup_scores = head_probs_all(head, reps_all)
del model, head; torch.cuda.empty_cache()

np.save(FEATURES_DIR / "ts_tcc_semisup_fulllabel_window_scores.npy", tstcc_semisup_scores)
log(f"  Saved: ts_tcc_semisup_fulllabel_window_scores.npy (for future reuse)")
sanity_thr = float(np.percentile(tstcc_semisup_scores[np.concatenate([val_ben_idx, val_lenc_idx])], FPR_PCT))
log(f"  Sanity check vs recorded seed=0 (thr=0.1832, 99% recall, 90% C06): this reproduction thr={sanity_thr:.4f}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Stratified 50/50 split, repeated, per head
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log(f"Non-circular FPR — {N_SPLITS} stratified 50/50 splits per head"); log("=" * 72)

HEADS = {
    "VAE_nonran_only": vae_nonran_scores,
    "TS_TCC_unsup":     tstcc_unsup_scores,
    "TS_TCC_semisup":   tstcc_semisup_scores,
    "BiLSTM":           lstm_scores_all,
}

def verdict(i, head_scores, thr):
    i = int(i)
    if pran[i] >= 0.5: return True
    # streaming omitted — proven exactly 0.0 on this exact eval set (item 63)
    return head_scores[i] > thr

json_out = {}
for head_name, scores in HEADS.items():
    log(f"\n  ── {head_name} ──")
    split_results = []
    for split_seed in range(N_SPLITS):
        rng = np.random.RandomState(split_seed)
        ben_shuf  = rng.permutation(val_ben_idx)
        lenc_shuf = rng.permutation(val_lenc_idx)
        halfA = np.concatenate([ben_shuf[:45], lenc_shuf[:45]])
        halfB_ben, halfB_lenc = ben_shuf[45:], lenc_shuf[45:]

        thr = float(np.percentile(scores[halfA], FPR_PCT))

        b_fp  = sum(1 for i in halfB_ben  if verdict(i, scores, thr))
        le_fp = sum(1 for i in halfB_lenc if verdict(i, scores, thr))
        n_variant = sum(1 for i in variant_idx if verdict(i, scores, thr))
        n_c06     = sum(1 for i in conti06_idx if verdict(i, scores, thr))

        split_results.append({
            "thr": thr,
            "fpr_ben": b_fp / len(halfB_ben), "fpr_lenc": le_fp / len(halfB_lenc),
            "fpr_combined": (b_fp + le_fp) / (len(halfB_ben) + len(halfB_lenc)),
            "variant_recall": n_variant / len(variant_idx),
            "conti06_recall": n_c06 / len(conti06_idx) if len(conti06_idx) else None,
        })

    def ms(key):
        vals = np.array([r[key] for r in split_results])
        return float(vals.mean()), float(vals.std())

    fb_m, fb_s = ms("fpr_ben"); fl_m, fl_s = ms("fpr_lenc"); fc_m, fc_s = ms("fpr_combined")
    vr_m, vr_s = ms("variant_recall"); c6_m, c6_s = ms("conti06_recall")
    log(f"    Held-out FPR (half B, {N_SPLITS} splits): Ben={fb_m:.1%}±{fb_s:.1%}  "
        f"LEnc={fl_m:.1%}±{fl_s:.1%}  Combined={fc_m:.1%}±{fc_s:.1%}")
    log(f"    Variant recall={vr_m:.1%}±{vr_s:.1%}  Conti_06={c6_m:.1%}±{c6_s:.1%}")
    h = HEADLINE[head_name]
    log(f"    Headline (full 180-exec calibration): Recall={h['recall']:.0%}  "
        f"C06={h['c06']:.0%}  FPR={h['fpr']:.1%}")
    json_out[head_name] = {
        "fpr_ben_mean": fb_m, "fpr_ben_std": fb_s, "fpr_lenc_mean": fl_m, "fpr_lenc_std": fl_s,
        "fpr_combined_mean": fc_m, "fpr_combined_std": fc_s,
        "variant_recall_mean": vr_m, "variant_recall_std": vr_s,
        "conti06_recall_mean": c6_m, "conti06_recall_std": c6_s,
        "headline": h, "per_split": split_results,
    }

log()
log("=" * 72); log("READ"); log("=" * 72)
log("  Held-out FPR near 5% and recall near headline -> operating point is sound, cite this")
log("  as the non-circular measurement. Held-out FPR materially higher -> the true operating")
log("  point is looser than reported; restate the headline FPR.")
log("  Scope: this addresses calibration/measurement circularity only — NOT item 59's session-")
log("  exposure finding (both halves still contain same-session-sibling executions).")

out_txt  = FEATURES_DIR / "heldout_fpr_split_results.txt"
out_json = FEATURES_DIR / "heldout_fpr_split_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("heldout_fpr_split.py done.")
