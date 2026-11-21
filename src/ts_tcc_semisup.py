"""
ts_tcc_semisup.py — Semi-Supervised TS-TCC: Frozen SSL Encoder + Supervised Head
=======================================================================================
Per HANDOFF "TWEAK 2" (Arshad's final TS-TCC experiment, August 7, 2026).

Different question from the unsupervised comparison (ts_tcc_detection.py /
ts_tcc_fusion.py): does self-supervised (TS-TCC) PRETRAINING + a small
supervised head beat supervised-from-raw-features (the BiLSTM)? This is
SEMI-SUPERVISED (self-supervised representation + supervised head) — report
and interpret it separately from the pure-unsupervised comparison.

Pipeline:
  1. Pretrain TS-TCC self-supervised (temporal-preserving augmentation — the
     Tweak-1 winning config: magnitude-warp + mild time-warp + jitter, no
     segment-permutation) on the training pool, no labels. 3 seeds.
  2. FREEZE the encoder. Train a small supervised head (1-layer MLP,
     128->64->2) on top of the representations, ransomware vs non-
     ransomware, known/train classes only. Two versions:
       (a) full_label — all 1800 training-pool labels.
       (b) few_shot   — a stratified ~7.5% subsample of the training labels
                         (the true semi-supervised test — TS-TCC's claimed
                         strength in the paper's Section 5.2).
  3. Plug the head into the SAME fusion as VAE/BiLSTM/TS-TCC-unsupervised
     (frozen classifier_2way.pt primary + streaming + this head's
     P(ransomware) + binary rule). Threshold = 95th pct of validation
     non-ransomware head score (same discipline throughout the project).
  4. Evaluate blind: variant recall, Conti_06 per-exec, Benign/LEnc/combined
     FPR, mean ± std over 3 seeds. LOFO for the full_label head only (single
     seed — same compute trade-off documented in the earlier TS-TCC scripts;
     few-shot LOFO skipped as out of scope for this exploratory add-on).
  5. Head-to-head vs BiLSTM (99%/90%/100% LOFO), TS-TCC-unsupervised
     (93%±2%/50%±14%/85% LOFO, from TS_TCC_FUSION), VAE (89%/30%/70% LOFO).

★  CORE RULE: pretrain/train on known classes only; Conti_06 NEVER in
   training, threshold calibration, or head-training data; threshold from
   validation non-ransomware only; blind eval on ALL variants + LOFO; FPR
   always reported. Report this result as SEMI-SUPERVISED, not unsupervised.
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

FEATURES_DIR  = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT       = 95.0
STREAM_THRESH = 0.65
SEEDS         = [0, 1, 2]
FEWSHOT_FRAC  = 0.075   # ~7.5% of training labels, midpoint of the paper's "5-10%"

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]
LOFO_FAMILIES        = {
    "WannaCry": ["WannaCry"], "Ryuk": ["Ryuk"], "REvil": ["REvil"],
    "LockBit": ["LockBit"], "Darkside": ["Darkside"], "Conti": ["Conti"],
}

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
JITTER_SIGMA = 0.03
SCALE_RATIO  = 2.0

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

val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
c06_idx        = [int(i) for i in variant_idx if all_labels[i] == "Conti_06"]

log(f"  Training pool: {len(train_idx)} (ran={train_y_bin.sum()}, non-ran={(train_y_bin==0).sum()})")
log(f"  Val: {len(val_idx)}  Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}")
log(f"  Variants: {len(variant_idx)},  Conti_06: {len(c06_idx)}")

scaler = StandardScaler()
scaler.fit(feat_23[train_idx].reshape(-1, 23))

def scale_seqs(seqs):
    N, T, C = seqs.shape
    return (scaler.transform(seqs.reshape(-1, C)).astype(np.float32)
            .reshape(N, T, C))

feat_23_scaled = scale_seqs(feat_23)

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Frozen 2-way CNN — probs + streaming (identical to ts_tcc_fusion.py)
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
# SECTION 3 — TS-TCC model defs + temporal-preserving augmentation (Tweak-1 winner)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("TS-TCC model definitions (temporal-preserving config only)"); log("=" * 72)

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


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Supervised head on frozen representations
# ════════════════════════════════════════════════════════════════════════════════

class SupervisedHead(nn.Module):
    def __init__(self, in_dim=128, hidden=64, n_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)

def train_head(reps_train, y_train, seed, n_epochs=100, lr=1e-3, batch_size=32):
    torch.manual_seed(seed); np.random.seed(seed)
    head = SupervisedHead(in_dim=reps_train.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    n_ran = y_train.sum(); n_rest = len(y_train) - n_ran
    w = torch.tensor([n_ran / len(y_train), n_rest / len(y_train)], dtype=torch.float32).to(device)
    X_t = torch.tensor(reps_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    dl = DataLoader(TensorDataset(X_t, y_t), min(batch_size, len(y_train)), shuffle=True)
    for ep in range(n_epochs):
        head.train()
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = head(bx)
            F.cross_entropy(logits, by, weight=w).backward()
            opt.step()
    return head

@torch.no_grad()
def head_probs_all(head, reps_all, batch_size=256):
    head.eval()
    probs = []
    for i in range(0, len(reps_all), batch_size):
        xb = torch.tensor(reps_all[i:i + batch_size], dtype=torch.float32).to(device)
        probs.append(torch.softmax(head(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

def stratified_subsample(idx_arr, y_bin, frac, seed):
    rng = np.random.RandomState(seed)
    idx_pos = idx_arr[y_bin == 1]
    idx_neg = idx_arr[y_bin == 0]
    n_pos = max(1, int(round(len(idx_pos) * frac)))
    n_neg = max(1, int(round(len(idx_neg) * frac)))
    sel_pos = rng.choice(idx_pos, n_pos, replace=False)
    sel_neg = rng.choice(idx_neg, n_neg, replace=False)
    return np.concatenate([sel_pos, sel_neg])


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Fusion verdict (identical structure to ts_tcc_fusion.py)
# ════════════════════════════════════════════════════════════════════════════════

def binary_verdict(i, thr, scores, pmap, smap):
    i = int(i)
    p = pmap[i]; s = smap.get(i, 0.0)
    fires = scores[i] > thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN RUN: full_label + few_shot heads, in-fusion, 3 seeds
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("MAIN RUN — semi-supervised TS-TCC (temporal-preserving), in-fusion, 3 seeds"); log("=" * 72)

X_train_scaled = feat_23_scaled[train_idx]
main_results = {"full_label": [], "few_shot": []}

for seed in SEEDS:
    t0 = time.time()
    log(f"\n  --- Seed {seed}: pretraining TS-TCC (temporal-preserving) ---")
    model = pretrain_tstcc(X_train_scaled, seed, tag=f"pretrain seed={seed}")
    reps_all = extract_representations(model, feat_23_scaled)
    log(f"  Pretraining done in {time.time()-t0:.1f}s")

    for head_type in ["full_label", "few_shot"]:
        if head_type == "full_label":
            head_train_idx = train_idx
            y_head = train_y_bin
        else:
            head_train_idx = stratified_subsample(train_idx, train_y_bin, FEWSHOT_FRAC, seed)
            y_head = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in head_train_idx])

        head = train_head(reps_all[head_train_idx], y_head, seed)
        scores = head_probs_all(head, reps_all)
        thr = float(np.percentile(scores[val_nonran_idx], FPR_PCT))

        verdicts_var = [binary_verdict(i, thr, scores, pmap2, smap2) for i in variant_idx]
        verdicts_c06 = [binary_verdict(i, thr, scores, pmap2, smap2) for i in c06_idx]
        verdicts_ben = [binary_verdict(i, thr, scores, pmap2, smap2) for i in val_ben_idx]
        verdicts_le  = [binary_verdict(i, thr, scores, pmap2, smap2) for i in val_lenc_idx]

        recall     = sum(v == "Attack" for v in verdicts_var) / len(variant_idx)
        c06_recall = sum(v == "Attack" for v in verdicts_c06) / len(c06_idx)
        fpr_ben    = sum(v == "Attack" for v in verdicts_ben) / len(val_ben_idx)
        fpr_lenc   = sum(v == "Attack" for v in verdicts_le) / len(val_lenc_idx)
        fpr_nr     = (sum(v == "Attack" for v in verdicts_ben) + sum(v == "Attack" for v in verdicts_le)) / len(val_nonran_idx)

        main_results[head_type].append({
            "n_labels": len(head_train_idx), "thr": thr, "recall": recall, "c06_recall": c06_recall,
            "fpr_ben": fpr_ben, "fpr_lenc": fpr_lenc, "fpr_nr": fpr_nr,
        })
        log(f"    [{head_type}, n_labels={len(head_train_idx)}] seed={seed}  thr={thr:.4f}  "
            f"recall={recall:.0%}  C06={c06_recall:.0%}  FPR_Ben={fpr_ben:.1%}  FPR_LEnc={fpr_lenc:.1%}  FPR_NR={fpr_nr:.1%}")

    del model; torch.cuda.empty_cache()

def summarize(metric_list, key):
    vals = np.array([m[key] for m in metric_list])
    return float(vals.mean()), float(vals.std())

log(f"\n  {'='*72}")
log("  MAIN RUN SUMMARY — SEMI-SUPERVISED, IN-FUSION (mean ± std over 3 seeds)")
log(f"  {'='*72}")
for head_type in ["full_label", "few_shot"]:
    rec_m, rec_s = summarize(main_results[head_type], "recall")
    c06_m, c06_s = summarize(main_results[head_type], "c06_recall")
    fb_m,  fb_s  = summarize(main_results[head_type], "fpr_ben")
    fl_m,  fl_s  = summarize(main_results[head_type], "fpr_lenc")
    fn_m,  fn_s  = summarize(main_results[head_type], "fpr_nr")
    log(f"  [{head_type:>11}]  Recall={rec_m:.0%}±{rec_s:.0%}  C06={c06_m:.0%}±{c06_s:.0%}  "
        f"FPR_Ben={fb_m:.1%}±{fb_s:.1%}  FPR_LEnc={fl_m:.1%}±{fl_s:.1%}  FPR_NR={fn_m:.1%}±{fn_s:.1%}")

log(f"\n  Reference — BiLSTM (supervised, raw feats) = 99% / 90% C06 / 100% LOFO")
log(f"              TS-TCC-unsupervised (Maha, temporal-preserving) = 93%±2% / 50%±14% C06 / 85% LOFO")
log(f"              VAE (unsupervised) = 89% / 30% C06 / 70% LOFO")

json_out["main_run_semisup"] = main_results


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LOFO (full_label head only, single seed=0 per fold)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("6-FAMILY LOFO — semi-supervised (full_label head), single seed per fold"); log("=" * 72)

X_tr_text = text_seqs[train_idx]

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

@torch.no_grad()
def probs_and_stream_batch(model, idx_list):
    model.eval()
    pmap, smap = {}, {}
    for i in idx_list:
        i = int(i)
        pmap[i] = probs_2way(model, text_seqs, i)
        smap[i] = 0.0 if pmap[i].argmax() == 1 else streaming_max_2way(model, text_seqs, i)
    return pmap, smap

lofo_results = {}
for family, members in LOFO_FAMILIES.items():
    t0 = time.time()
    log(f"\n  === LOFO: hold out {family} ===")
    held_mask   = np.array([all_labels[i] in members for i in train_idx])
    held_global = train_idx[np.where(held_mask)[0]]
    fold_tr_idx = train_idx[np.where(~held_mask)[0]]
    fold_tr_y2  = train_y_bin[np.where(~held_mask)[0]]

    # Fold CNN (3-seed pick-best, same procedure as ts_tcc_fusion.py)
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

    # Fold TS-TCC + head (single seed=0)
    model = pretrain_tstcc(feat_23_scaled[fold_tr_idx], seed=0, tag=f"LOFO-{family}")
    reps_all = extract_representations(model, feat_23_scaled)
    head = train_head(reps_all[fold_tr_idx], fold_tr_y2, seed=0)
    scores = head_probs_all(head, reps_all)
    thr = float(np.percentile(scores[val_nonran_idx], FPR_PCT))

    pmap_held, smap_held = probs_and_stream_batch(best_clf, [int(i) for i in held_global])
    pmap_nr,   smap_nr   = probs_and_stream_batch(best_clf, [int(i) for i in val_nonran_idx])

    n_att = sum(1 for i in held_global if binary_verdict(i, thr, scores, pmap_held, smap_held) == "Attack")
    b_fp  = sum(1 for i in val_ben_idx  if binary_verdict(i, thr, scores, pmap_nr, smap_nr) == "Attack")
    le_fp = sum(1 for i in val_lenc_idx if binary_verdict(i, thr, scores, pmap_nr, smap_nr) == "Attack")

    n_held = len(held_global)
    recall = n_att / n_held
    log(f"    N={n_held}  Attack={n_att}  Recall={recall:.0%}  "
        f"FPR_Ben={b_fp/len(val_ben_idx):.1%}  FPR_LEnc={le_fp/len(val_lenc_idx):.1%}  ({time.time()-t0:.0f}s)")

    lofo_results[family] = {
        "n": n_held, "attack": n_att, "recall": recall,
        "fpr_ben": b_fp / len(val_ben_idx), "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": (b_fp + le_fp) / len(val_nonran_idx),
    }
    del model, head, best_clf; torch.cuda.empty_cache()

json_out["lofo_semisup_full_label"] = lofo_results

log(f"\n  LOFO SUMMARY — semi-supervised (full_label head), single seed")
log(f"  {'Family':<12}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
log("  " + "─" * 54)
recalls = []
for fam, r in lofo_results.items():
    log(f"  {fam:<12}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  "
        f"{r['fpr_lenc']:>10.1%}  {r['fpr_nr']:>8.1%}")
    recalls.append(r["recall"])
log(f"  {'MEAN':<12}  {np.mean(recalls):>8.0%}")
log(f"\n  Reference — BiLSTM LOFO: 100% all families. TS-TCC-unsupervised LOFO: 85-86%. VAE LOFO: 70%.")


# ════════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════════
out_txt  = FEATURES_DIR / "ts_tcc_semisup_results.txt"
out_json = FEATURES_DIR / "ts_tcc_semisup_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("ts_tcc_semisup.py done.")
