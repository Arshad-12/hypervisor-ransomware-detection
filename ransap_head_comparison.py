"""
ransap_head_comparison.py — Cross-Dataset RanSAP Evaluation as a Head Comparison
================================================================================================
Per HANDOFF item 61 (August 16, 2026).

RANSAP_CROSS_DATASET (July 15, job 17697336) reports cross-dataset transfer with a SINGLE head
(storage-only 5-dim BiLSTM). Every other results subsection in this project compares multiple
heads at a matched operating point; this extends the cross-dataset subsection to match, adding
TS-TCC (self-supervised, storage-only) and a storage-only VAE alongside the existing BiLSTM.

★ Hard constraint: RanSAP has no memory logs, so EVERY head here is storage-only (5-dim: T_sr,
  T_sw, V_sr, V_sw, H_sw). This is a property of the dataset, not a design choice — the result is
  NOT a memory ablation and must not be presented as one (it changes OS, hardware, families, and
  modality all at once vs the RanSMAP-internal results).

Pipeline held fixed, only the anomaly head swapped:
  frozen classifier_2way.pt (trained on RanSMAP, MiniLM text) + MiniLM streaming (shared) +
  {BiLSTM | TS-TCC | VAE} storage-only head + binary fusion (classifier -> streaming -> head).

  - BiLSTM (5-dim): REUSED from RANSAP_CROSS_DATASET, NOT re-run. Numbers pulled directly from
    Results.md / job 17697336's log. ★ RECATEGORIZED here: AESCrypt is BENIGN in RanSAP (item-46
    correction) — the July table incorrectly listed AESCrypt's 93.3% "recall" in the ransomware
    table; it belongs in the FPR table (93.3% FALSE POSITIVE rate on the AESCrypt legit-encryption
    tool), which this script's summary table corrects. This is a re-tabulation of already-computed
    numbers, not a rerun.
    ★ CAVEAT — threshold protocol difference (stated explicitly, not glossed over): the July
    BiLSTM run calibrated its threshold on RanSMAP TRAINING non-ransomware scores (see
    `ransap_eval.py`'s `train_bilstm`), not RanSMAP validation. TS-TCC and VAE below are
    calibrated on RanSMAP VALIDATION non-ransomware, per this item's stated discipline. The three
    heads are therefore NOT calibrated on an identical calibration set — flagged here rather than
    presented as a clean three-way match.
  - TS-TCC (5-dim): NEW. Encoder pretrained self-supervised on RanSMAP storage-only 5-dim
    sequences (temporal-preserving augmentation, the winning in-project config), single seed=0.
    Mahalanobis distance-to-normal head fit on RanSMAP non-ransomware training representations.
    Threshold = 95th pct of RanSMAP VALIDATION non-ransomware TS-TCC score.
  - VAE (5-dim): NEW (optional third point, included since cheap — small MLP). Same
    TimeSeriesVAE architecture as `vae_nonran_only.py`, adapted to 5-dim input, trained on
    RanSMAP non-ransomware-only training subset. Threshold = 95th pct of RanSMAP VALIDATION
    non-ransomware MSE.

Both new heads' StandardScaler is fit on RanSMAP only, then applied to RanSAP features
unchanged (same convention as the existing BiLSTM cross-dataset run) — trained on RanSMAP only,
RanSAP evaluated fully blind.

Reports per head: per-family recall over the 7 real ransomware families (TeslaCrypt, Cerber,
WannaCry, Ryuk, GandCrab4, Sodinokibi, Darkside), FPR on standard benign (Firefox, Excel)
separate from encryption-tool FPR (SDelete, Zip, AESCrypt) — matching the July run's split.
Notes whether Cerber (H_sw=0.052, an information-floor case per CERBER_DIAGNOSIS) is missed by
every head or only some.
"""

import json, re, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

RANSAP_FEAT  = Path("/scratch/aabdulj4/RanSAP/features")
RANSMAP_FEAT = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT       = 95.0
STREAM_THRESH = 0.65
CNN_THRESH    = 0.5
N_WINDOWS     = 300
FEAT_DIM      = 5
TSTCC_SEED    = 0
VAE_SEED      = 0
VAE_EPOCHS    = 80

RANSMAP_RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
RANSMAP_LEGIT_ENC_CLASSES  = ["AESCrypt", "SDelete", "Zip"]
RANSMAP_BENIGN_CLASSES     = ["Firefox", "Office", "Idle"]
RANSMAP_ALL_ORIGINAL       = RANSMAP_RANSOMWARE_CLASSES + RANSMAP_LEGIT_ENC_CLASSES + RANSMAP_BENIGN_CLASSES

# RanSAP label conventions (from ransap_eval.py, confirmed against actual directory listing)
RAN_LABELS    = ["Cerber", "Darkside", "GandCrab4", "Ryuk", "Sodinokibi", "TeslaCrypt", "WannaCry"]
BENIGN_LABELS = ["AESCrypt", "Excel", "Firefox", "SDelete", "Zip"]
HIGH_ENT_BENIGN = {"AESCrypt", "SDelete", "Zip"}

# BiLSTM reference numbers — REUSED from RANSAP_CROSS_DATASET (job 17697336), NOT recomputed.
# Recall table EXCLUDES AESCrypt (item-46 correction: AESCrypt is benign in RanSAP).
BILSTM_REF_RECALL = {
    "Cerber": 11/140, "Darkside": 140/140, "GandCrab4": 124/140, "Ryuk": 163/163,
    "Sodinokibi": 266/310, "TeslaCrypt": 126/141, "WannaCry": 159/160,
}
BILSTM_REF_N = {
    "Cerber": 140, "Darkside": 140, "GandCrab4": 140, "Ryuk": 163,
    "Sodinokibi": 310, "TeslaCrypt": 141, "WannaCry": 160,
}
# FPR table, AESCrypt moved here from the (incorrect) July recall table.
BILSTM_REF_FPR = {
    "Firefox": 1/60, "Excel": 6/61,               # standard benign
    "SDelete": 5/60, "Zip": 53/60, "AESCrypt": 56/60,  # encryption-tool / high-entropy
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
# SECTION 1 — Load RanSMAP (train the heads) + RanSAP (blind eval)
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72); log("Loading RanSMAP + RanSAP data"); log("=" * 72)

smap_meta   = pd.read_csv(RANSMAP_FEAT / "metadata.csv")
smap_labels = np.load(RANSMAP_FEAT / "text_labels.npy", allow_pickle=True).astype(str)
smap_text_seqs = np.load(RANSMAP_FEAT / "text_sequences.npy")

split      = np.load(RANSMAP_FEAT / "fewshot_split_cnn.npz", allow_pickle=True)
train_idx  = split["train_idx"]
val_idx    = split["val_idx"]

log("  Loading RanSMAP feat_23 (need only columns 0:5, storage-only)...")
t0 = time.time()
smap_feat23 = np.zeros((len(smap_meta), N_WINDOWS, 23), dtype=np.float32)
for ii, row in smap_meta.iterrows():
    fp = row.get("filepath", row.get("feat_path", None))
    if fp and Path(fp).exists():
        smap_feat23[ii] = np.load(fp)
smap_feat5 = smap_feat23[:, :, :FEAT_DIM]
del smap_feat23
log(f"  RanSMAP feat5: {smap_feat5.shape}  ({time.time()-t0:.1f}s)")

def is_smap_ran(lbl):
    return any(lbl == r or lbl.startswith(r + "_") or lbl.startswith(r + "-") for r in RANSMAP_RANSOMWARE_CLASSES)
def is_smap_nonran_known(lbl):
    return lbl in RANSMAP_ALL_ORIGINAL and not is_smap_ran(lbl)

train_ran_smap_idx    = np.array([i for i in train_idx if is_smap_ran(smap_labels[i])])
train_nonran_smap_idx = np.array([i for i in train_idx if is_smap_nonran_known(smap_labels[i])])
val_nonran_smap_idx   = np.array([i for i in val_idx if is_smap_nonran_known(smap_labels[i])])
log(f"  RanSMAP train: ran={len(train_ran_smap_idx)}, non-ran={len(train_nonran_smap_idx)}  "
    f"val non-ran={len(val_nonran_smap_idx)}")

# RanSAP
rsp_feat5      = np.load(RANSAP_FEAT / "ransap_feat5.npy")
rsp_text_seqs  = np.load(RANSAP_FEAT / "ransap_text_seqs.npy")
rsp_labels_raw = np.load(RANSAP_FEAT / "ransap_labels.npy", allow_pickle=True).astype(str)

def norm_label(lbl):
    return re.sub(r'-(largefiles|w10dirs.*)$', '', lbl)
rsp_labels = np.array([norm_label(l) for l in rsp_labels_raw])

rsp_role = np.full(len(rsp_labels), "unknown", dtype=object)
for i, lbl in enumerate(rsp_labels):
    if lbl in RAN_LABELS: rsp_role[i] = "ransomware"
    elif lbl in BENIGN_LABELS: rsp_role[i] = "benign"

rsp_active = np.where(rsp_role != "unknown")[0]
log(f"  RanSAP: {rsp_feat5.shape}  active execs (known role): {len(rsp_active)}")

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Frozen classifier_2way.pt: classifier + streaming on RanSAP
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Frozen classifier_2way.pt: classifier + streaming (no retrain)"); log("=" * 72)

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

cnn_clf = TextEmbeddingClassifier(emb_dim=384, n_classes=2).to(device)
cnn_clf.load_state_dict(torch.load(RANSMAP_FEAT / "classifier_2way.pt", map_location=device))
cnn_clf.eval()

@torch.no_grad()
def cnn_scores(text_seqs, indices, batch=64):
    idx_list = list(indices); all_scores = []
    for b in range(0, len(idx_list), batch):
        bidx = idx_list[b:b+batch]
        X = np.stack([text_seqs[int(i)] for i in bidx])
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        logits, _, _ = cnn_clf(Xt)
        all_scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(all_scores)

@torch.no_grad()
def streaming_scores(text_seqs, indices):
    W = 50; scores = []
    for i in indices:
        i = int(i); seq = text_seqs[i]; best = 0.0
        for start in range(0, N_WINDOWS - W + 1, 5):
            chunk  = seq[start:start + W]
            padded = np.concatenate([chunk, np.zeros((N_WINDOWS - W, text_seqs.shape[-1]), np.float32)])
            x = torch.tensor(padded[None], dtype=torch.float32).to(device)
            logits, _, _ = cnn_clf(x)
            p = torch.softmax(logits, dim=1)[0, 1].item()
            if p > best: best = p
        scores.append(best)
    return np.array(scores, dtype=np.float32)

p_cnn_rsp    = cnn_scores(rsp_text_seqs, rsp_active)
log("  Streaming (only for execs where classifier didn't already fire)...")
s_stream_rsp = np.zeros(len(rsp_active), dtype=np.float32)
need_stream  = np.where(p_cnn_rsp < CNN_THRESH)[0]
if len(need_stream):
    s_stream_rsp[need_stream] = streaming_scores(rsp_text_seqs, rsp_active[need_stream])
log(f"  Done. Classifier fired on {int((p_cnn_rsp >= CNN_THRESH).sum())}/{len(rsp_active)} RanSAP execs alone.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TS-TCC (5-dim, storage-only) — pretrain on RanSMAP, score RanSAP blind
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("TS-TCC (5-dim storage-only) — pretrain on RanSMAP, blind eval on RanSAP"); log("=" * 72)

scaler5 = StandardScaler()
scaler5.fit(smap_feat5[train_idx].reshape(-1, FEAT_DIM))
def scale5(seqs):
    N, T, C = seqs.shape
    return scaler5.transform(seqs.reshape(-1, C)).astype(np.float32).reshape(N, T, C)
smap_feat5_scaled = scale5(smap_feat5)
rsp_feat5_scaled  = scale5(rsp_feat5)

JITTER_SIGMA, SCALE_RATIO = 0.03, 2.0
TRANS_H, TRANS_HEADS, TRANS_LAYERS = 100, 4, 4
LAMBDA1, LAMBDA2, TAU = 1.0, 0.7, 0.2
N_EPOCHS, BATCH_SIZE, LR, WD = 40, 128, 3e-4, 3e-4

class Encoder(nn.Module):
    def __init__(self, in_channels=FEAT_DIM, hidden=128, dropout=0.35):
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
    def __init__(self, in_channels=FEAT_DIM, enc_hidden=128, K=16, h=TRANS_H):
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
    for i in range(B):
        out[i] = x_np[i] * _smooth_curve(T, n_knots, sigma, loc=1.0)[:, None]
    return out

def time_warp_np(x_np, sigma=0.05, n_knots=4):
    B, T, C = x_np.shape; out = np.empty_like(x_np); orig_steps = np.arange(T)
    for i in range(B):
        curve = np.clip(_smooth_curve(T, n_knots, sigma, loc=1.0), 1e-3, None)
        cum = np.cumsum(curve); cum = (cum - cum.min()) / (cum.max() - cum.min() + 1e-8) * (T - 1)
        for c in range(C):
            out[i, :, c] = np.interp(orig_steps, cum, x_np[i, :, c])
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
        if (ep+1) % 20 == 0: log(f"    epoch {ep+1}/{n_epochs}  loss={ep_loss/n_batches:.4f}")
    return model

@torch.no_grad()
def extract_representations(model, X_scaled, batch_size=256):
    model.eval(); reps = []
    for i in range(0, len(X_scaled), batch_size):
        xb = torch.tensor(X_scaled[i:i+batch_size], dtype=torch.float32).to(device)
        reps.append(model.encoder(xb).mean(dim=-1).cpu().numpy())
    return np.concatenate(reps, axis=0)

t0 = time.time()
tstcc_model = pretrain_tstcc(smap_feat5_scaled[train_idx], seed=TSTCC_SEED)
reps_smap = extract_representations(tstcc_model, smap_feat5_scaled)
reps_rsp  = extract_representations(tstcc_model, rsp_feat5_scaled)
log(f"  TS-TCC pretrain + extraction done in {time.time()-t0:.0f}s")

reps_normal_ref = reps_smap[train_nonran_smap_idx]
mu = reps_normal_ref.mean(axis=0)
VI = np.linalg.pinv(np.cov(reps_normal_ref.T))
def maha_score(reps):
    diff = reps - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, VI, diff))

tstcc_score_smap = maha_score(reps_smap)
tstcc_score_rsp  = maha_score(reps_rsp)
tstcc_thresh = float(np.percentile(tstcc_score_smap[val_nonran_smap_idx], FPR_PCT))
log(f"  TS-TCC threshold (95th pct RanSMAP VAL non-ran, n={len(val_nonran_smap_idx)}): {tstcc_thresh:.4f}")
del tstcc_model; torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VAE (5-dim, storage-only, non-ransomware-only) — pretrain on RanSMAP
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("VAE (5-dim storage-only, non-ransomware-only) — pretrain on RanSMAP"); log("=" * 72)

class TimeSeriesVAE5(nn.Module):
    def __init__(self, seq_len=300, n_features=FEAT_DIM, latent_dim=16):
        super().__init__()
        D = seq_len * n_features
        self.enc = nn.Sequential(
            nn.Linear(D, 512),   nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 512),        nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, D))
        self.seq_len, self.n_features = seq_len, n_features
    def encode(self, x):
        h = self.enc(x.view(x.size(0), -1)); return self.fc_mu(h), self.fc_logvar(h)
    def decode(self, z): return self.dec(z).view(z.size(0), self.seq_len, self.n_features)
    def forward(self, x):
        mu, logvar = self.encode(x); z = mu + torch.exp(0.5*logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

def vae_loss_fn(recon, x, mu, logvar, beta=1.0):
    mse = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return mse + beta * kld

def train_vae5(X_scaled, n_epochs=VAE_EPOCHS, lr=1e-3, batch_size=64, seed=VAE_SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TimeSeriesVAE5().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    dl = DataLoader(TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(n_epochs):
        for (bx,) in dl:
            bx = bx.to(device); opt.zero_grad()
            recon, mu, logvar = model(bx)
            vae_loss_fn(recon, bx, mu, logvar).backward(); opt.step()
        sched.step()
    return model

@torch.no_grad()
def vae_mse_batch(model, X_scaled):
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)), batch_size=64, shuffle=False)
    mse_list = []
    for (bx,) in dl:
        bx = bx.to(device); mu, _ = model.encode(bx); recon = model.decode(mu)
        mse_list.append(F.mse_loss(recon, bx, reduction="none").mean(dim=[1, 2]).cpu().numpy())
    return np.concatenate(mse_list)

vae_scaler5 = StandardScaler()
vae_scaler5.fit(smap_feat5[train_nonran_smap_idx].reshape(-1, FEAT_DIM))
def vae_scale5(seqs):
    N, T, C = seqs.shape
    return vae_scaler5.transform(seqs.reshape(-1, C)).astype(np.float32).reshape(N, T, C)

smap_feat5_vaescaled = vae_scale5(smap_feat5)
rsp_feat5_vaescaled  = vae_scale5(rsp_feat5)

t0 = time.time()
vae_model = train_vae5(smap_feat5_vaescaled[train_nonran_smap_idx])
vae_score_smap = vae_mse_batch(vae_model, smap_feat5_vaescaled)
vae_score_rsp  = vae_mse_batch(vae_model, rsp_feat5_vaescaled)
log(f"  VAE train + score done in {time.time()-t0:.0f}s")
vae_thresh = float(np.percentile(vae_score_smap[val_nonran_smap_idx], FPR_PCT))
log(f"  VAE threshold (95th pct RanSMAP VAL non-ran, n={len(val_nonran_smap_idx)}): {vae_thresh:.4f}")
del vae_model; torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Fusion + report per head
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("HEAD COMPARISON — RanSAP cross-dataset, per-family recall + FPR"); log("=" * 72)

def binary_verdict(p, s, head_score, thr):
    if p >= CNN_THRESH: return True
    if s >= STREAM_THRESH: return True
    return head_score > thr

def report_head(name, head_scores_active, thresh):
    ran_families = sorted(set(rsp_labels[rsp_active]) & set(RAN_LABELS))
    recall = {}
    for fam in ran_families:
        fam_mask = rsp_labels[rsp_active] == fam
        n = fam_mask.sum()
        det = sum(1 for j in np.where(fam_mask)[0]
                   if binary_verdict(p_cnn_rsp[j], s_stream_rsp[j], head_scores_active[j], thresh))
        recall[fam] = {"n": int(n), "det": int(det), "recall": det / n if n else None}
    fpr = {}
    for fam in ["Firefox", "Excel", "SDelete", "Zip", "AESCrypt"]:
        fam_mask = rsp_labels[rsp_active] == fam
        n = fam_mask.sum()
        if n == 0: continue
        fp = sum(1 for j in np.where(fam_mask)[0]
                  if binary_verdict(p_cnn_rsp[j], s_stream_rsp[j], head_scores_active[j], thresh))
        fpr[fam] = {"n": int(n), "fp": int(fp), "fpr": fp / n if n else None}
    log(f"\n  ── {name}  (threshold={thresh:.4f}) ──")
    log(f"  {'Family':<14} {'N':>5} {'Det':>5} {'Recall':>8}")
    for fam, r in recall.items():
        log(f"  {fam:<14} {r['n']:>5} {r['det']:>5} {r['recall']:>7.1%}")
    log(f"  FPR — standard benign:")
    for fam in ["Firefox", "Excel"]:
        if fam in fpr: log(f"    {fam:<12} {fpr[fam]['n']:>5} {fpr[fam]['fp']:>5} {fpr[fam]['fpr']:>7.1%}")
    log(f"  FPR — encryption-tool / high-entropy:")
    for fam in ["SDelete", "Zip", "AESCrypt"]:
        if fam in fpr: log(f"    {fam:<12} {fpr[fam]['n']:>5} {fpr[fam]['fp']:>5} {fpr[fam]['fpr']:>7.1%}")
    return {"recall": recall, "fpr": fpr}

json_out["tstcc"] = report_head("TS-TCC (5-dim, NEW)", tstcc_score_rsp[rsp_active], tstcc_thresh)
json_out["vae"]   = report_head("VAE (5-dim, NEW)",    vae_score_rsp[rsp_active],   vae_thresh)

log(f"\n  ── BiLSTM (5-dim, REUSED from RANSAP_CROSS_DATASET job 17697336, NOT rerun) ──")
log(f"  {'Family':<14} {'N':>5} {'Recall':>8}")
for fam, r in BILSTM_REF_RECALL.items():
    log(f"  {fam:<14} {BILSTM_REF_N[fam]:>5} {r:>7.1%}")
log(f"  FPR — standard benign: Firefox {BILSTM_REF_FPR['Firefox']:.1%}  Excel {BILSTM_REF_FPR['Excel']:.1%}")
log(f"  FPR — encryption-tool: SDelete {BILSTM_REF_FPR['SDelete']:.1%}  Zip {BILSTM_REF_FPR['Zip']:.1%}  "
    f"AESCrypt {BILSTM_REF_FPR['AESCrypt']:.1%}  (recategorized from July's recall table — item-46 correction)")
json_out["bilstm_reused"] = {"recall": BILSTM_REF_RECALL, "fpr": BILSTM_REF_FPR}

log()
log("=" * 72); log("CERBER — missed by every head, or only some?"); log("=" * 72)
log(f"  BiLSTM:  {BILSTM_REF_RECALL['Cerber']:.1%}")
log(f"  TS-TCC:  {json_out['tstcc']['recall']['Cerber']['recall']:.1%}")
log(f"  VAE:     {json_out['vae']['recall']['Cerber']['recall']:.1%}")

out_txt  = RANSMAP_FEAT / "ransap_head_comparison_results.txt"
out_json = RANSMAP_FEAT / "ransap_head_comparison_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("ransap_head_comparison.py done.")
