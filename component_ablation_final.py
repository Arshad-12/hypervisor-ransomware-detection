"""
component_ablation_final.py — Component Ablation on the Final Pipeline (professor rev #3)
================================================================================================
Per HANDOFF item 58, TASK A (August 15, 2026).

The only ablation on record (NB15, `15_ablation_final_figures.ipynb`) ran locally, pre-feat_23-fix,
pre-2-way-fusion, on small-scale data (25% combined FPR, VAE trained on 96 execs) — not comparable
with the final pipeline and must not go in the report. This script produces the real one: three
decision rules (classifier alone / +streaming / +streaming+anomaly-head), evaluated per anomaly
head (VAE non-ransomware-only, TS-TCC unsupervised, BiLSTM), on the LOCKED final pipeline.

★ No retraining of the classifier or the VAE/BiLSTM heads — all three reuse existing frozen
  artifacts on disk:
    classifier_2way.pt              (frozen 2-way CNN, shared everywhere)
    vae_nonran_window_scores.npy    (VAE non-ransomware-only, from vae_nonran_only.py)
    lstm_scores_all.npy             (BiLSTM, from binary_fusion.py — main-run/within-family head,
                                      unaffected by the item-57 LOFO leak since this is not LOFO)
  ONE exception: no prior script persisted per-exec TS-TCC-unsupervised scores to disk (only
  summary stats were logged in TS_TCC_FUSION). This script reproduces that exact winning config
  (temporal_preserving augmentation + Mahalanobis head, single seed=0 — matching the per-seed-0
  row in TS_TCC_FUSION: 94% recall / 60% Conti_06) ONCE, purely to capture per-exec scores, and
  saves them (ts_tcc_unsup_window_scores.npy) so this doesn't need to happen again. This is NOT a
  new experiment or a retrain of a different config — it is a single reproduction of the existing
  best result for the purpose of scoring individual executions.

For each rule × head, threshold = 95th pct of that head's score on validation non-ransomware
(recomputed fresh here for all three heads, so the ablation is self-consistent rather than
depending on three different scripts' possibly-different threshold-saving conventions).

Reports: variant recall (all Conti_01-07, ~70 execs), Conti_06 alone, FPR_Ben, FPR_LEnc, FPR_NR.

★ Cross-family (LOFO) version of this ablation was NOT run — none of the prior LOFO scripts
  persisted per-fold per-exec signals to disk (only summary stats), so "if the per-fold signals
  are still available" (HANDOFF item 58) resolves to: they are not. Noted here explicitly rather
  than silently skipped.
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

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT       = 95.0
STREAM_THRESH = 0.65

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES

# TS-TCC winning config (from TS_TCC_FUSION) — reproduced once, single seed=0
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
TSTCC_SEED   = 0

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
# SECTION 1 — Load data + existing artifacts
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72); log("Loading data + existing frozen artifacts"); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
meta_df    = pd.read_csv(FEATURES_DIR / "metadata.csv")
text_seqs  = np.load(FEATURES_DIR / "text_sequences.npy")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]

val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
val_nonran_idx = np.concatenate([val_ben_idx, val_lenc_idx])
conti06_idx    = np.array([i for i in variant_idx if all_labels[i] == "Conti_06"])

log(f"  Variants: {len(variant_idx)}  (Conti_06: {len(conti06_idx)})")
log(f"  Val non-ran: {len(val_nonran_idx)} (Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)})")

log("\n  Loading feat_23 (needed only for the TS-TCC reproduction run)...")
t0 = time.time()
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])
log(f"  Loaded in {time.time()-t0:.1f}s")

# ── Frozen classifier_2way.pt ───────────────────────────────────────────────────
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
log("\n  classifier_2way.pt loaded (frozen, no retraining).")

@torch.no_grad()
def probs_2way(i):
    x = torch.tensor(text_seqs[int(i)][None], dtype=torch.float32).to(device)
    logits, _, _ = clf2(x)
    return torch.softmax(logits, dim=1).cpu().numpy()[0]

@torch.no_grad()
def streaming_max_2way(i):
    seq = text_seqs[int(i)]; W = 50; best = 0.0
    for start in range(0, 300 - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = clf2(x)
        p = torch.softmax(logits, dim=1)[0, 1].item()
        if p > best: best = p
    return best

eval_idx = np.concatenate([variant_idx, val_idx])
log(f"  Precomputing classifier probs + streaming for {len(eval_idx)} execs (variants + val)...")
pmap, smap = {}, {}
for i in eval_idx:
    i = int(i)
    p = probs_2way(i)
    pmap[i] = p[1]  # P(ransomware)
    smap[i] = 0.0 if p[1] >= 0.5 else streaming_max_2way(i)
log("  Done.")

# ── VAE non-ransomware-only (existing artifact, no retrain) ────────────────────
vae_nonran_scores = np.load(FEATURES_DIR / "vae_nonran_window_scores.npy")
log(f"\n  Loaded vae_nonran_window_scores.npy ({vae_nonran_scores.shape}) — no retrain.")

# ── BiLSTM main-run (existing artifact, no retrain — unaffected by item-57 leak) ─
lstm_scores_all = np.load(FEATURES_DIR / "lstm_scores_all.npy")
log(f"  Loaded lstm_scores_all.npy ({lstm_scores_all.shape}) — no retrain, main-run head "
    f"(trained on train_idx only, same as always; item-57's leak was cross-family/LOFO-specific).")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TS-TCC unsupervised: ONE reproduction run (winning config, seed=0)
# to capture per-exec scores (no prior script saved these to disk)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72)
log("TS-TCC unsupervised — single reproduction of TS_TCC_FUSION's winning config")
log("(temporal_preserving aug + Mahalanobis, seed=0) to capture per-exec scores")
log("=" * 72)

scaler = StandardScaler()
scaler.fit(feat_23[train_idx].reshape(-1, 23))
def scale_seqs(seqs):
    N, T, C = seqs.shape
    return scaler.transform(seqs.reshape(-1, C)).astype(np.float32).reshape(N, T, C)
feat_23_scaled = scale_seqs(feat_23)

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

def pretrain_tstcc(X_scaled, seed, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE):
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
            log(f"    epoch {ep+1}/{n_epochs}  loss={ep_loss/n_batches:.4f}")
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

t0 = time.time()
tstcc_model = pretrain_tstcc(feat_23_scaled[train_idx], seed=TSTCC_SEED)
reps_all = extract_representations(tstcc_model, feat_23_scaled)

train_nonran_mask = np.array([all_labels[i] not in RANSOMWARE_CLASSES for i in train_idx])
reps_normal_ref = reps_all[train_idx][train_nonran_mask]
mu = reps_normal_ref.mean(axis=0)
VI = np.linalg.pinv(np.cov(reps_normal_ref.T))
diff = reps_all - mu
tstcc_unsup_scores = np.sqrt(np.einsum("ij,jk,ik->i", diff, VI, diff))
log(f"  TS-TCC reproduction done in {time.time()-t0:.0f}s.")

np.save(FEATURES_DIR / "ts_tcc_unsup_window_scores.npy", tstcc_unsup_scores)
log("  Saved: ts_tcc_unsup_window_scores.npy (for future reuse — no prior script had this).")
del tstcc_model; torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Three decision rules × three heads
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("COMPONENT ABLATION — three rules x three anomaly heads"); log("=" * 72)

HEADS = {
    "VAE_nonran_only": vae_nonran_scores,
    "TS_TCC_unsup":     tstcc_unsup_scores,
    "BiLSTM":           lstm_scores_all,
}

def eval_rule(rule, head_scores=None, head_thr=None):
    """rule in {'classifier', 'classifier_stream', 'classifier_stream_head'}"""
    def verdict(i):
        i = int(i)
        p = pmap[i]; s = smap[i]
        if p >= 0.5: return True
        if rule == "classifier": return False
        if s >= STREAM_THRESH: return True
        if rule == "classifier_stream": return False
        return head_scores[i] > head_thr
    n_variant = sum(1 for i in variant_idx if verdict(i))
    n_c06     = sum(1 for i in conti06_idx if verdict(i))
    b_fp      = sum(1 for i in val_ben_idx  if verdict(i))
    le_fp     = sum(1 for i in val_lenc_idx if verdict(i))
    return {
        "variant_recall": n_variant / len(variant_idx),
        "conti06_recall": n_c06 / len(conti06_idx) if len(conti06_idx) else None,
        "fpr_ben":  b_fp / len(val_ben_idx),
        "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr":   (b_fp + le_fp) / len(val_nonran_idx),
    }

json_out = {}

# Rules 1 & 2 don't depend on the head — compute once
r1 = eval_rule("classifier")
r2 = eval_rule("classifier_stream")
log(f"\n  Rule 1 — Classifier alone:")
log(f"    Variant recall={r1['variant_recall']:.0%}  Conti_06={r1['conti06_recall']:.0%}  "
    f"FPR_Ben={r1['fpr_ben']:.1%}  FPR_LEnc={r1['fpr_lenc']:.1%}  FPR_NR={r1['fpr_nr']:.1%}")
log(f"  Rule 2 — Classifier + streaming:")
log(f"    Variant recall={r2['variant_recall']:.0%}  Conti_06={r2['conti06_recall']:.0%}  "
    f"FPR_Ben={r2['fpr_ben']:.1%}  FPR_LEnc={r2['fpr_lenc']:.1%}  FPR_NR={r2['fpr_nr']:.1%}")
json_out["rule1_classifier_alone"] = r1
json_out["rule2_classifier_streaming"] = r2
json_out["rule3_per_head"] = {}

for head_name, scores in HEADS.items():
    thr = float(np.percentile(scores[val_nonran_idx], FPR_PCT))
    r3 = eval_rule("classifier_stream_head", head_scores=scores, head_thr=thr)
    log(f"\n  Rule 3 — Classifier + streaming + {head_name}  (thr={thr:.4f}):")
    log(f"    Variant recall={r3['variant_recall']:.0%}  Conti_06={r3['conti06_recall']:.0%}  "
        f"FPR_Ben={r3['fpr_ben']:.1%}  FPR_LEnc={r3['fpr_lenc']:.1%}  FPR_NR={r3['fpr_nr']:.1%}")
    log(f"    Incremental gain over classifier+streaming: "
        f"variant {r2['variant_recall']:.0%}->{r3['variant_recall']:.0%}, "
        f"Conti_06 {r2['conti06_recall']:.0%}->{r3['conti06_recall']:.0%}")
    r3["thr"] = thr
    json_out["rule3_per_head"][head_name] = r3

# ── Summary table ──────────────────────────────────────────────────────────────
log(); log("=" * 72); log("SUMMARY — COMPONENT ABLATION, FINAL PIPELINE"); log("=" * 72)
log(f"  {'Rule':<38} {'Variant':>8} {'Conti06':>8} {'FPR_Ben':>9} {'FPR_LEnc':>10} {'FPR_NR':>8}")
log("  " + "-" * 85)
log(f"  {'Classifier alone':<38} {r1['variant_recall']:>7.0%} {r1['conti06_recall']:>7.0%} "
    f"{r1['fpr_ben']:>8.1%} {r1['fpr_lenc']:>9.1%} {r1['fpr_nr']:>7.1%}")
log(f"  {'Classifier + streaming':<38} {r2['variant_recall']:>7.0%} {r2['conti06_recall']:>7.0%} "
    f"{r2['fpr_ben']:>8.1%} {r2['fpr_lenc']:>9.1%} {r2['fpr_nr']:>7.1%}")
for head_name in HEADS:
    r3 = json_out["rule3_per_head"][head_name]
    label = f"+ {head_name}"
    log(f"  {label:<38} {r3['variant_recall']:>7.0%} {r3['conti06_recall']:>7.0%} "
        f"{r3['fpr_ben']:>8.1%} {r3['fpr_lenc']:>9.1%} {r3['fpr_nr']:>7.1%}")

log()
log("  NOTE: cross-family (LOFO) version of this ablation NOT run — no prior LOFO script")
log("  persisted per-fold per-exec signals to disk, so the per-fold data needed does not exist.")

(FEATURES_DIR / "component_ablation_final_results.txt").write_text("\n".join(lines))
with open(FEATURES_DIR / "component_ablation_final_results.json", "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(); log("Saved: component_ablation_final_results.txt / .json")
log("component_ablation_final.py done.")
