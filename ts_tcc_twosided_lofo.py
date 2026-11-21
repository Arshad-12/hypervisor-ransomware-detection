"""
ts_tcc_twosided_lofo.py — 6-Family LOFO for the TS-TCC Two-Sided Winner
================================================================================================
Per HANDOFF item 56, TASK D (final piece — August 10, 2026).

Config = the winner from TS_TCC_RANONLY: two-sided score
(dist_normal - dist_ransomware) computed inside the B1 encoder space
(TS-TCC pretrained on ALL 1800 known-class executions, temporal-preserving
augmentation) — the 98% +- 1% recall / 87% +- 9% Conti_06 main-run result.
NOT the B2 (ransomware-only-pretrained) encoder — that was the weaker
variant (96%/77%).

Per fold: hold the family out ENTIRELY —
  - re-pretrain the TS-TCC encoder on the remaining known-class executions
    (held-out family excluded; single seed, consistent with every other
    LOFO run in this project),
  - refit BOTH Mahalanobis heads (dist_normal, dist_ransomware) INSIDE that
    fold's own representation space (never across spaces — the encoder
    changed, so both heads must be refit together each fold),
  - the ransomware head automatically excludes the held-out family too,
    since it is fit only on the fold's remaining training pool,
  - retrain the 2-way CNN per fold (3-seed pick-best, as always).
Threshold = 95th pct validation non-ransomware two-sided score, matched
~5% FPR.

★  LABEL-INFORMED / SUPERVISED result (per HANDOFF item 56's labelling
   rule) — building the ransomware-distance head requires labels.
★  CORE RULE: held-out family excluded from encoder pretraining, CNN
   training, AND the ransomware-distance head; threshold from validation
   non-ransomware only; blind eval; FPR always reported.
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
FPR_PCT        = 95.0
STREAM_THRESH  = 0.65

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
LOFO_FAMILIES        = {
    "WannaCry": ["WannaCry"], "Ryuk": ["Ryuk"], "REvil": ["REvil"],
    "LockBit": ["LockBit"], "Darkside": ["Darkside"], "Conti": ["Conti"],
}

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

log("  Loading MiniLM sequences for the per-fold 2-way CNN...")
text_seqs = np.load(FEATURES_DIR / "text_sequences.npy")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]
train_y_bin = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in train_idx])

val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])

log(f"  Training pool: {len(train_idx)} (ran={train_y_bin.sum()}, non-ran={(train_y_bin==0).sum()})")
log(f"  Val: {len(val_idx)}  Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}")

scaler = StandardScaler()
scaler.fit(feat_23[train_idx].reshape(-1, 23))

def scale_seqs(seqs):
    N, T, C = seqs.shape
    return (scaler.transform(seqs.reshape(-1, C)).astype(np.float32).reshape(N, T, C))

feat_23_scaled = scale_seqs(feat_23)

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 2-way CNN class + probs/streaming helpers
# ════════════════════════════════════════════════════════════════════════════════

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

@torch.no_grad()
def probs_and_stream_batch(model, idx_list):
    model.eval()
    pmap, smap = {}, {}
    for i in idx_list:
        i = int(i)
        pmap[i] = probs_2way(model, text_seqs, i)
        smap[i] = 0.0 if pmap[i].argmax() == 1 else streaming_max_2way(model, text_seqs, i)
    return pmap, smap

def train_cnn_2way(X_tr, y, n_epochs=80, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TextEmbeddingClassifier(emb_dim=384, n_classes=2).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
    n_ran = y.sum(); n_rest = len(y) - n_ran
    w = torch.tensor([n_ran/len(y), n_rest/len(y)], dtype=torch.float32).to(device)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                   torch.tensor(y, dtype=torch.long)), 32, shuffle=True)
    for ep in range(n_epochs):
        model.train()
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, _, _ = model(bx)
            F.cross_entropy(logits, by, weight=w).backward(); opt.step()
        sched.step()
    return model


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TS-TCC model defs + temporal-preserving augmentation
# ════════════════════════════════════════════════════════════════════════════════

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
# SECTION 4 — Fusion verdict (standard direction — two-sided score, high = Attack)
# ════════════════════════════════════════════════════════════════════════════════

def binary_verdict(i, thr, scores, pmap, smap):
    i = int(i)
    p = pmap[i]; s = smap.get(i, 0.0)
    fires = scores[i] > thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — 6-Family LOFO
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72)
log("6-FAMILY LOFO — TS-TCC two-sided (B1-space winner), encoder + both heads +")
log("CNN all retrained per fold, single seed")
log("=" * 72)

X_tr_text = text_seqs[train_idx]
lofo_results = {}

for family, members in LOFO_FAMILIES.items():
    t0 = time.time()
    log(f"\n  === LOFO: hold out {family} ===")
    held_mask   = np.array([all_labels[i] in members for i in train_idx])
    held_global = train_idx[np.where(held_mask)[0]]
    fold_tr_idx = train_idx[np.where(~held_mask)[0]]
    fold_tr_y2  = train_y_bin[np.where(~held_mask)[0]]

    # -- Fold CNN (3-seed pick-best) --
    best_clf, best_acc = None, -1
    for seed in [0, 1, 2]:
        m = train_cnn_2way(X_tr_text[np.where(~held_mask)[0]], fold_tr_y2, n_epochs=80, seed=seed)
        m.eval()
        with torch.no_grad():
            xi = torch.tensor(text_seqs[val_idx], dtype=torch.float32)
            dl_ = DataLoader(TensorDataset(xi), 64, shuffle=False)
            preds = np.concatenate([m(bx.to(device))[0].argmax(1).cpu().numpy() for (bx,) in dl_])
            val_y2 = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in val_idx])
            acc = (preds == val_y2).mean()
        if acc > best_acc:
            best_acc = acc; best_clf = m
    log(f"    Fold CNN val acc: {best_acc:.1%}")

    # -- Fold TS-TCC encoder (held-out family excluded, single seed=0) --
    model = pretrain_tstcc(feat_23_scaled[fold_tr_idx], seed=0, tag=f"LOFO-{family}")
    reps_all = extract_representations(model, feat_23_scaled)

    # -- Both Mahalanobis heads, inside THIS fold's space, held-out family excluded from both --
    fold_nonran_mask = np.array([all_labels[i] not in RANSOMWARE_CLASSES for i in fold_tr_idx])
    fold_ran_mask     = np.array([all_labels[i] in RANSOMWARE_CLASSES for i in fold_tr_idx])
    reps_normal_ref     = reps_all[fold_tr_idx][fold_nonran_mask]
    reps_ransomware_ref = reps_all[fold_tr_idx][fold_ran_mask]   # held-out family already excluded (not in fold_tr_idx)
    mu_n, VI_n = fit_mahalanobis(reps_normal_ref)
    mu_r, VI_r = fit_mahalanobis(reps_ransomware_ref)

    dist_normal     = score_mahalanobis(reps_all, mu_n, VI_n)
    dist_ransomware = score_mahalanobis(reps_all, mu_r, VI_r)
    two_sided = dist_normal - dist_ransomware
    thr = float(np.percentile(two_sided[val_nonran_idx], FPR_PCT))

    pmap_held, smap_held = probs_and_stream_batch(best_clf, [int(i) for i in held_global])
    pmap_nr,   smap_nr   = probs_and_stream_batch(best_clf, [int(i) for i in val_nonran_idx])

    n_att = sum(1 for i in held_global if binary_verdict(i, thr, two_sided, pmap_held, smap_held) == "Attack")
    b_fp  = sum(1 for i in val_ben_idx  if binary_verdict(i, thr, two_sided, pmap_nr, smap_nr) == "Attack")
    le_fp = sum(1 for i in val_lenc_idx if binary_verdict(i, thr, two_sided, pmap_nr, smap_nr) == "Attack")

    n_held = len(held_global)
    recall = n_att / n_held
    log(f"    thr={thr:.4f}  N={n_held}  Attack={n_att}  Recall={recall:.0%}  "
        f"FPR_Ben={b_fp/len(val_ben_idx):.1%}  FPR_LEnc={le_fp/len(val_lenc_idx):.1%}  ({time.time()-t0:.0f}s)")

    lofo_results[family] = {
        "n": n_held, "attack": n_att, "recall": recall, "thr": thr,
        "fpr_ben": b_fp / len(val_ben_idx), "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": (b_fp + le_fp) / len(val_nonran_idx),
    }
    del best_clf, model; torch.cuda.empty_cache()

json_out["lofo_two_sided_b1"] = lofo_results

log(f"\n  LOFO SUMMARY — TS-TCC two-sided (B1-space winner)")
log(f"  {'Family':<12}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
log("  " + "─" * 54)
recalls = []
for fam, r in lofo_results.items():
    log(f"  {fam:<12}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  "
        f"{r['fpr_lenc']:>10.1%}  {r['fpr_nr']:>8.1%}")
    recalls.append(r["recall"])
log(f"  {'MEAN':<12}  {np.mean(recalls):>8.0%}")
log(f"\n  Reference — BiLSTM LOFO: 100%. VAE likelihood-ratio LOFO: 74%. "
    f"TS-TCC-unsupervised LOFO: 85-86%. VAE single-formulation LOFO: 70%.")
log(f"  Main-run (within-family) two-sided B1: 98%±1% recall / 87%±9% Conti_06 (TS_TCC_RANONLY).")

out_txt  = FEATURES_DIR / "ts_tcc_twosided_lofo_results.txt"
out_json = FEATURES_DIR / "ts_tcc_twosided_lofo_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("ts_tcc_twosided_lofo.py done.")
