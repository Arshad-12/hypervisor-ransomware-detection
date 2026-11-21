"""
mpnet_full_fusion.py — Fair mpnet Full Pipeline Evaluation (v2, fixed streaming)
=================================================================================
Pure inference — NO training. Uses already-saved models:
  mpnet_classifier.pt          — trained in Priority 1
  vae_fixed_model.pt + scaler  — trained in D3-FIXED
  vae_fixed_threshold.npy
  lstm_scores_all.npy          — saved by binary_fusion.py (load if available)

Streaming fix (v2):
  PROBLEM: Removing zero-padding in v1 broke MiniLM streaming (60% FPR).
  Root cause: 50-window unpadded inputs are OOD for a CNN trained on 300-window
  sequences. The mean-pool sees a very different distribution at 50 vs 300 steps.

  The ORIGINAL MiniLM streaming (zero-padded to 300) was correct — it produced
  84%/1.7% FPR in the 16d baseline. Zero-padding adds inactive windows that
  dilute the active windows and keep the distribution in-range for MiniLM.

  For mpnet: zero-padding also breaks it (100% FPR), because zero vectors
  (non-natural mpnet embeddings) hit BatchNorm and push P(ran) to ~1.

  FIX: MiniLM streaming uses ORIGINAL zero-padded approach (restoring v0).
       mpnet fusion uses MiniLM streaming signal (smap_minilm) — fair ablation:
       only the PRIMARY classifier changes; streaming + VAE/BiLSTM stay identical.

Evaluates:
  v4_minilm_vae    — MiniLM primary + MiniLM streaming (zero-pad) + fixed VAE
  v4_mpnet_hybrid  — mpnet primary  + MiniLM streaming (zero-pad) + fixed VAE
  v6_minilm_bilstm — MiniLM primary + MiniLM streaming (zero-pad) + BiLSTM (if available)
  v6_mpnet_hybrid  — mpnet primary  + MiniLM streaming (zero-pad) + BiLSTM (if available)

Saves: mpnet_full_fusion_results.txt / .json
"""

import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
STREAM_THRESH = 0.65
BENIGN_IDX, LENC_IDX, RAN_IDX = 0, 1, 2

RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES     = ["Firefox", "Office", "Idle"]
VARIANT_CLASSES    = [f"Conti_{i:02d}" for i in range(1, 8)]

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
# SECTION 1 — Load data + splits
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72); log("Loading data"); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
split      = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx    = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx    = np.where(split["mix_mask"])[0]
eval_idx   = np.concatenate([val_idx, variant_idx, mix_idx])

val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ran_idx    = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])
c06_idx        = [int(i) for i in variant_idx if all_labels[i] == "Conti_06"]

log(f"  Val: {len(val_idx)} (ben={len(val_ben_idx)}, lenc={len(val_lenc_idx)}, "
    f"ran={len(val_ran_idx)})")
log(f"  Variants: {len(variant_idx)},  Conti_06: {len(c06_idx)}")

# Load VAE artifacts (from D3-FIXED)
log("\n  Loading VAE fixed artifacts...")
from sklearn.preprocessing import StandardScaler
vae_scaler = StandardScaler()
vae_scaler.mean_  = np.load(FEATURES_DIR / "vae_fixed_scaler_mean.npy")
vae_scaler.scale_ = np.load(FEATURES_DIR / "vae_fixed_scaler_scale.npy")
vae_scaler.n_features_in_ = 23
vae_thresh = float(np.load(FEATURES_DIR / "vae_fixed_threshold.npy")[0])
log(f"  VAE threshold: {vae_thresh:.4f}")

# Load BiLSTM scores if available (from binary_fusion.py)
lstm_scores_path = FEATURES_DIR / "lstm_scores_all.npy"
if lstm_scores_path.exists():
    lstm_scores_all = np.load(lstm_scores_path)
    lstm_thresh     = float(np.load(FEATURES_DIR / "lstm_threshold.npy")[0])
    log(f"  Loaded lstm_scores_all.npy  (thresh={lstm_thresh:.4f})")
    has_lstm = True
else:
    log("  lstm_scores_all.npy not found — BiLSTM fusion columns will be skipped")
    has_lstm = False


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VAE model + score all 1970 execs
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("VAE fixed — scoring all 1970 execs"); log("=" * 72)

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
    def decode(self, z): return self.dec(z).view(z.size(0), 300, 23)
    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.decode(z), mu, lv

vae = TimeSeriesVAE().to(device)
vae.load_state_dict(torch.load(FEATURES_DIR / "vae_fixed_model.pt",
                               map_location=device, weights_only=True))
vae.eval()

log("  Loading feat_23 for all 1970 execs...")
meta_df = pd.read_csv(FEATURES_DIR / "metadata.csv")
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])

def scale_seqs(seqs):
    N, T, nf = seqs.shape
    return (vae_scaler.transform(seqs.reshape(-1, nf).astype(np.float64))
            .astype(np.float32).reshape(N, T, nf))

@torch.no_grad()
def vae_mse_batch(X_scaled):
    dl = DataLoader(TensorDataset(torch.tensor(X_scaled)), 64, shuffle=False)
    mse = []
    for (bx,) in dl:
        bx = bx.to(device); mu, _ = vae.encode(bx)
        recon = vae.decode(mu)
        mse.append(F.mse_loss(recon, bx, reduction="none").mean(dim=[1,2]).cpu().numpy())
    return np.concatenate(mse)

log("  Scaling + scoring with VAE...")
X_all_scaled = scale_seqs(feat_23)
vae_mse_all  = vae_mse_batch(X_all_scaled)
log(f"  VAE MSE: val non-ran mean={vae_mse_all[val_nonran_idx].mean():.4f}  "
    f"val ran mean={vae_mse_all[val_ran_idx].mean():.4f}")
log(f"  Val FPR (VAE alone): {float((vae_mse_all[val_nonran_idx] > vae_thresh).mean()):.1%}")
del feat_23, X_all_scaled


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CNN classifiers + streaming
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("CNN classifiers + streaming"); log("=" * 72)
log("  Streaming: ORIGINAL zero-padded approach for MiniLM (restoring baseline).")
log("  mpnet fusions will REUSE smap_minilm — fair primary-classifier ablation.")

class TextEmbeddingClassifier(nn.Module):
    def __init__(self, emb_dim=384, n_classes=3):
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

# Load MiniLM (3-way)
log("  Loading MiniLM sequences (1970, 300, 384)...")
minilm_seqs = np.load(FEATURES_DIR / "text_sequences.npy")
clf_minilm  = TextEmbeddingClassifier(emb_dim=384, n_classes=3).to(device)
clf_minilm.load_state_dict(torch.load(FEATURES_DIR / "classifier.pt",
                                      map_location=device, weights_only=True))
clf_minilm.eval()

# Load mpnet (3-way)
log("  Loading mpnet sequences (1970, 300, 768)...")
mpnet_seqs = np.load(FEATURES_DIR / "mpnet_sequences.npy")
clf_mpnet  = TextEmbeddingClassifier(emb_dim=768, n_classes=3).to(device)
clf_mpnet.load_state_dict(torch.load(FEATURES_DIR / "mpnet_classifier.pt",
                                     map_location=device, weights_only=True))
clf_mpnet.eval()

@torch.no_grad()
def primary_probs(clf, seqs, i):
    x = torch.tensor(seqs[int(i)][None], dtype=torch.float32).to(device)
    logits, _, _ = clf(x)
    return torch.softmax(logits, dim=1).cpu().numpy()[0]

@torch.no_grad()
def streaming_zeropads(clf, seqs, i, emb_dim):
    """ORIGINAL zero-padded streaming — correct for MiniLM.
    50-window slice is padded to 300 with zeros; CNN mean-pools over all 300
    positions (diluting active windows). This is what produced 84%/1.7% in 16d.
    Do NOT use for mpnet: zero vectors are unnatural mpnet embeddings and push
    P(ran) to ~1 via BatchNorm artefacts."""
    seq = seqs[int(i)]; W = 50; best = 0.0
    for start in range(0, 300 - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((300 - W, emb_dim), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = clf(x)
        p = torch.softmax(logits, dim=1)[0, RAN_IDX].item()
        if p > best: best = p
    return best

log(f"  Precomputing primary probs + MiniLM streaming for {len(eval_idx)} execs...")
pmap_minilm, smap_minilm = {}, {}
pmap_mpnet  = {}

for ii, i in enumerate(eval_idx):
    i = int(i)
    pmap_minilm[i] = primary_probs(clf_minilm, minilm_seqs, i)
    pmap_mpnet[i]  = primary_probs(clf_mpnet,  mpnet_seqs,  i)
    # MiniLM streaming: only compute for non-ran-primary execs
    if pmap_minilm[i].argmax() != RAN_IDX:
        smap_minilm[i] = streaming_zeropads(clf_minilm, minilm_seqs, i, 384)
    else:
        smap_minilm[i] = 0.0
    if (ii + 1) % 200 == 0: log(f"    {ii+1}/{len(eval_idx)}")

log("  Done.")

# Sanity check: streaming FPR should be low (~0-5%) for Benign execs
s_nonran = [smap_minilm[int(i)] for i in val_nonran_idx]
s_ben    = [smap_minilm[int(i)] for i in val_ben_idx]
s_lenc   = [smap_minilm[int(i)] for i in val_lenc_idx]
log(f"\n  MiniLM streaming sanity (SHOULD match baseline, ~0% Ben, ~0-5% LEnc):")
log(f"    val Benign  — mean={np.mean(s_ben):.3f}   ≥0.65: {np.mean(np.array(s_ben) >= STREAM_THRESH):.1%}")
log(f"    val LEnc    — mean={np.mean(s_lenc):.3f}   ≥0.65: {np.mean(np.array(s_lenc) >= STREAM_THRESH):.1%}")
log(f"    val non-ran — mean={np.mean(s_nonran):.3f}   ≥0.65: {np.mean(np.array(s_nonran) >= STREAM_THRESH):.1%}")

del minilm_seqs, mpnet_seqs


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Fusion evaluation
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("FUSION — mpnet vs MiniLM primary (shared streaming + VAE)"); log("=" * 72)

def fuse_v4(p, s, vae_mse):
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]: return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        if s >= STREAM_THRESH or vae_mse > vae_thresh: return "Attack"
        return "Benign"
    if s >= STREAM_THRESH: return "Attack"
    if vae_mse > vae_thresh: return "Idle"
    return "Benign"

def fuse_v6_bilstm(p, s, vae_mse, i):
    """3-way primary + BiLSTM head (shared stream). Only if lstm scores available."""
    if not has_lstm: return fuse_v4(p, s, vae_mse)
    score = lstm_scores_all[int(i)]
    fires = score > lstm_thresh
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]: return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"
    if s >= STREAM_THRESH: return "Attack"
    if fires: return "Idle"
    return "Benign"

def get_verdict(fusion, i):
    i = int(i)
    s = smap_minilm[i]           # always use MiniLM streaming
    v = vae_mse_all[i]
    if fusion == "v4_minilm_vae":
        return fuse_v4(pmap_minilm[i], s, v)
    if fusion == "v4_mpnet_hybrid":
        return fuse_v4(pmap_mpnet[i], s, v)
    if fusion == "v6_minilm_bilstm":
        return fuse_v6_bilstm(pmap_minilm[i], s, v, i)
    if fusion == "v6_mpnet_hybrid":
        return fuse_v6_bilstm(pmap_mpnet[i], s, v, i)

fusions = ["v4_minilm_vae", "v4_mpnet_hybrid"]
if has_lstm:
    fusions += ["v6_minilm_bilstm", "v6_mpnet_hybrid"]

json_out = {}

# ── Val FPR ──────────────────────────────────────────────────────────────────
log(f"\n── VAL FPR ──")
log(f"  {'Fusion':>22}  {'Ben FPR':>9}  {'LEnc FPR':>10}  {'NR FPR':>8}")
log("  " + "─" * 56)
for fname in fusions:
    b_fp  = sum(1 for i in val_ben_idx  if get_verdict(fname, i) == "Attack")
    le_fp = sum(1 for i in val_lenc_idx if get_verdict(fname, i) == "Attack")
    nr_fp = b_fp + le_fp
    log(f"  {fname:>22}  {b_fp}/{len(val_ben_idx)} ({b_fp/len(val_ben_idx):.1%})  "
        f"{le_fp}/{len(val_lenc_idx)} ({le_fp/len(val_lenc_idx):.1%})  "
        f"{nr_fp}/{len(val_nonran_idx)} ({nr_fp/len(val_nonran_idx):.1%})")
    json_out.setdefault("val_fpr", {})[fname] = {
        "fpr_ben": b_fp/len(val_ben_idx), "fpr_lenc": le_fp/len(val_lenc_idx),
        "fpr_nr": nr_fp/len(val_nonran_idx),
    }

# ── Val Recall (known classes) ────────────────────────────────────────────────
log(f"\n── VAL RECALL (known ransomware classes) ──")
log(f"  {'Fusion':>22}  {'Recall':>8}")
for fname in fusions:
    n_att = sum(1 for i in val_ran_idx if get_verdict(fname, i) == "Attack")
    log(f"  {fname:>22}  {n_att}/{len(val_ran_idx)} ({n_att/len(val_ran_idx):.1%})")
    json_out.setdefault("val_recall", {})[fname] = n_att / len(val_ran_idx)

# ── Variant recall ────────────────────────────────────────────────────────────
log(f"\n── VARIANT RECALL ──")
for fname in fusions:
    log(f"\n  [{fname}]")
    log(f"  {'Variant':<14}  {'N':>4}  {'Attack':>7}  {'Idle':>6}  {'Benign':>7}  {'Recall':>8}")
    log(f"  {'─'*50}")
    total_att = 0; per_v = {}
    for vname in VARIANT_CLASSES:
        vi = [i for i in variant_idx if all_labels[i] == vname]
        n_att = sum(1 for i in vi if get_verdict(fname, i) == "Attack")
        n_idle= sum(1 for i in vi if get_verdict(fname, i) == "Idle")
        n_ben = sum(1 for i in vi if get_verdict(fname, i) == "Benign")
        total_att += n_att
        flag = " ◀" if vname == "Conti_06" else ""
        log(f"  {vname+flag:<14}  {len(vi):>4}  {n_att:>7}  {n_idle:>6}  {n_ben:>7}  "
            f"{n_att/len(vi):>8.0%}")
        per_v[vname] = {"n": len(vi), "attack": n_att, "idle": n_idle,
                        "benign": n_ben, "recall": n_att/len(vi)}
    log(f"  {'ALL':<14}  {len(variant_idx):>4}  {total_att:>7}  {'':>6}  {'':>7}  "
        f"{total_att/len(variant_idx):>8.0%}")
    json_out.setdefault("variant_recall", {})[fname] = {
        "overall": total_att/len(variant_idx), "per_variant": per_v}

# ── Conti_06 per-exec ─────────────────────────────────────────────────────────
log(f"\n── CONTI_06 PER-EXECUTION ──")
header = f"  {'Idx':>6}  {'minilm_p(ran)':>14}  {'mpnet_p(ran)':>13}  {'vae_mse':>8}  {'stream':>7}"
for f in fusions:
    header += f"  {f:>20}"
log(header)
for i in c06_idx:
    row = "  ".join(f"{get_verdict(f, i):>20}" for f in fusions)
    log(f"  {i:>6}  {pmap_minilm[i][RAN_IDX]:>14.4f}  {pmap_mpnet[i][RAN_IDX]:>13.4f}  "
        f"{vae_mse_all[i]:>8.4f}  {smap_minilm[i]:>7.4f}  {row}")

# ── Mix set ───────────────────────────────────────────────────────────────────
if len(mix_idx) > 0:
    log(f"\n── MIX SET (100 execs) ──")
    for fname in fusions:
        mix_att = sum(1 for i in mix_idx if get_verdict(fname, i) == "Attack")
        mix_idle= sum(1 for i in mix_idx if get_verdict(fname, i) == "Idle")
        log(f"  {fname:>22}: Attack={mix_att}  Idle={mix_idle}  Benign={len(mix_idx)-mix_att-mix_idle}")
        json_out.setdefault("mix", {})[fname] = {"attack": mix_att, "idle": mix_idle}

# ── Primary classifier accuracy on val ───────────────────────────────────────
log(f"\n── PRIMARY CLASSIFIER VAL ACCURACY ──")
for clf_name, pmap in [("MiniLM", pmap_minilm), ("mpnet", pmap_mpnet)]:
    correct = 0; total = len(val_idx)
    for i in val_idx:
        pred = pmap[int(i)].argmax()
        lbl  = all_labels[int(i)]
        true = RAN_IDX if lbl in RANSOMWARE_CLASSES else (LENC_IDX if lbl in LEGIT_ENC_CLASSES else BENIGN_IDX)
        if pred == true: correct += 1
    log(f"  {clf_name:>8}: {correct}/{total} ({correct/total:.1%})")
    json_out.setdefault("primary_acc", {})[clf_name] = correct / total

# ── Summary table ─────────────────────────────────────────────────────────────
log(); log("=" * 72); log("SUMMARY")
log(f"\n  Reference: MiniLM + BiLSTM head (D3-FIXED best) → 87% / 1.1% FPR")
log(f"  All mpnet fusions use MiniLM streaming (shared signal, fair ablation)")
log(f"\n  {'Fusion':>22}  {'V.Recall':>9}  {'NR FPR':>8}  {'Conti_06':>10}")
log("  " + "─" * 56)
for fname in fusions:
    rec = json_out["variant_recall"][fname]["overall"]
    fpr = json_out["val_fpr"][fname]["fpr_nr"]
    c06 = json_out["variant_recall"][fname]["per_variant"].get("Conti_06", {}).get("recall", 0.0)
    log(f"  {fname:>22}  {rec:>9.0%}  {fpr:>8.1%}  {c06:>10.0%}")

# ── Save ─────────────────────────────────────────────────────────────────────
(FEATURES_DIR / "mpnet_full_fusion_results.txt").write_text("\n".join(lines))
with open(FEATURES_DIR / "mpnet_full_fusion_results.json", "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log("\nSaved: mpnet_full_fusion_results.txt / .json")
log("mpnet_full_fusion.py done.")
