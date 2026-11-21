"""
RanSMAP Fusion v2 — Full-Scale (Alliance, 16d)
================================================
Tests both fusion rules on the full 1,800-execution dataset:

  Fusion v1 (OR):  Escalate Benign→Attack if streaming fires OR VAE fires
  Fusion v2 (AND): Escalate Benign→Attack if streaming fires AND VAE fires

Key question: does the two-signal AND rule cut the 25% FPR while
keeping ~100% Conti_06 recall, now that the VAE is trained on 1,800
samples (FPR@95 dropped from 81.9% → 15.3%)?

Saves: fusion_v2_results.txt, fusion_v2_results.json
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")

# ── Class definitions ─────────────────────────────────────────────────────────
RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]
CLASS_ORDER          = ["benign", "legit_encryption", "ransomware"]
BENIGN_IDX, LENC_IDX, RAN_IDX = 0, 1, 2

# ── Thresholds ────────────────────────────────────────────────────────────────
STREAM_THRESH = 0.65   # from NB14 optimal operating point
W_STREAM      = 50
STRIDE        = 5
N_WINDOWS     = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Models ────────────────────────────────────────────────────────────────────
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


class TimeSeriesVAE(nn.Module):
    def __init__(self, seq_len=300, n_features=23, latent_dim=16):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
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
        return self.dec(z).view(z.size(0), self.seq_len, self.n_features)
    def forward(self, x):
        mu, lv = self.encode(x)
        return self.decode(mu), mu, lv   # deterministic at eval


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
text_seqs  = np.load(FEATURES_DIR / "text_sequences.npy")   # (1970, 300, 384)
all_labels = np.load(FEATURES_DIR / "text_labels.npy")
all_groups = np.load(FEATURES_DIR / "text_groups.npy")

# Load 23-dim features
meta_df = pd.read_csv(FEATURES_DIR / "metadata.csv")
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])
print(f"Text seqs: {text_seqs.shape}  |  Features: {feat_23.shape}")

# Load split
split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_mask = split["variant_mask"]
mix_mask    = split["mix_mask"]
variant_idx = np.where(variant_mask)[0]
mix_idx     = np.where(mix_mask)[0]

print(f"Val: {len(val_idx)}  |  Variants: {len(variant_idx)}  |  Mix: {len(mix_idx)}")

# Load models
print("Loading classifier...")
clf = TextEmbeddingClassifier().to(device)
clf.load_state_dict(torch.load(FEATURES_DIR / "classifier.pt", map_location=device, weights_only=True))
clf.eval()

print("Loading VAE...")
vae = TimeSeriesVAE().to(device)
vae.load_state_dict(torch.load(FEATURES_DIR / "vae_model.pt", map_location=device, weights_only=True))
vae.eval()

vae_mean      = np.load(FEATURES_DIR / "vae_scaler_mean.npy")
vae_scale     = np.load(FEATURES_DIR / "vae_scaler_scale.npy")
vae_threshold = float(np.load(FEATURES_DIR / "vae_threshold.npy")[0])
print(f"VAE threshold (95th pct): {vae_threshold:.4f}")


# ── Inference helpers ─────────────────────────────────────────────────────────
def primary_probs_batch(indices):
    """Run primary classifier on a set of indices, return (N, 3) probabilities."""
    probs = np.zeros((len(indices), 3), dtype=np.float32)
    batch_size = 64
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        x = torch.tensor(text_seqs[batch_idx], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, _, _ = clf(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
        probs[start:start + len(batch_idx)] = p
    return probs


def streaming_max_prob(idx):
    """Max P(ransomware) over W=50 sliding windows for one execution."""
    seq = text_seqs[idx]   # (300, 384)
    best = 0.0
    clf.eval()
    with torch.no_grad():
        for start in range(0, N_WINDOWS - W_STREAM + 1, STRIDE):
            chunk = seq[start:start + W_STREAM]   # (50, 384)
            # Pad to 300
            pad = np.zeros((N_WINDOWS - W_STREAM, 384), dtype=np.float32)
            padded = np.concatenate([chunk, pad], axis=0)
            x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
            logits, _, _ = clf(x)
            p = torch.softmax(logits, dim=1)[0, RAN_IDX].item()
            if p > best:
                best = p
    return best


def vae_mse(idx):
    """VAE reconstruction MSE for one execution (deterministic mu)."""
    feat = feat_23[idx]   # (300, 23)
    feat_flat = feat.reshape(-1, 23)
    feat_scaled = ((feat_flat - vae_mean) / vae_scale).reshape(1, 300, 23).astype(np.float32)
    x = torch.tensor(feat_scaled).to(device)
    with torch.no_grad():
        mu, _ = vae.encode(x)
        recon = vae.decode(mu)
        mse = F.mse_loss(recon, x, reduction="mean").item()
    return mse


# ── Run inference on all eval sets ────────────────────────────────────────────
eval_indices = np.concatenate([val_idx, variant_idx, mix_idx])
print(f"\nRunning primary classifier on {len(eval_indices)} executions...")
all_eval_probs = primary_probs_batch(eval_indices)

# Map index → probs
probs_map = {int(idx): all_eval_probs[i] for i, idx in enumerate(eval_indices)}

# Streaming + VAE only needed when primary says Benign
print("Computing streaming and VAE for Benign-primary executions...")
stream_map = {}
vae_map    = {}
for idx in eval_indices:
    p = probs_map[int(idx)]
    if p.argmax() == BENIGN_IDX:   # primary leans Benign → need both signals
        stream_map[int(idx)] = streaming_max_prob(idx)
        vae_map[int(idx)]    = vae_mse(idx)
    elif p.argmax() == LENC_IDX:
        stream_map[int(idx)] = 0.0   # not consulted for legit-enc
        vae_map[int(idx)]    = 0.0
    else:
        stream_map[int(idx)] = 0.0
        vae_map[int(idx)]    = 0.0

print(f"  Streaming/VAE computed for {sum(1 for i in eval_indices if probs_map[int(i)].argmax()==BENIGN_IDX)} Benign-primary executions")


# ── Fusion rules ──────────────────────────────────────────────────────────────
def fuse_v1(p, stream, vae_err):
    """OR rule: escalate if streaming OR VAE fires (replicated from local NB10)."""
    if p[LENC_IDX] >= p[RAN_IDX] and p[LENC_IDX] >= p[BENIGN_IDX]:
        return "Benign"   # Rule 1: protect legit-enc
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]:
        return "Attack"   # Rule 2: trust ransomware call
    # Rule 3: Benign-primary — consult both signals (OR)
    if stream >= STREAM_THRESH:
        return "Attack"
    if 0 < vae_err < 1e6 and vae_err > vae_threshold:
        return "Idle"
    return "Benign"


def fuse_v2(p, stream, vae_err):
    """AND rule: escalate ONLY when BOTH streaming AND VAE fire."""
    if p[LENC_IDX] >= p[RAN_IDX] and p[LENC_IDX] >= p[BENIGN_IDX]:
        return "Benign"
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]:
        return "Attack"
    # Rule 3: Benign-primary — require BOTH to escalate
    vae_fires    = 0 < vae_err < 1e6 and vae_err > vae_threshold
    stream_fires = stream >= STREAM_THRESH
    if stream_fires and vae_fires:
        return "Attack"
    if stream_fires or vae_fires:
        return "Idle"   # one signal fires → suspicious but not confirmed
    return "Benign"


def classify_only(p):
    idx = p.argmax()
    if idx == RAN_IDX:  return "Attack"
    if idx == LENC_IDX: return "Benign"
    return "Benign"


# Apply all three rules
results = {}
for idx in eval_indices:
    i = int(idx)
    p = probs_map[i]
    s = stream_map[i]
    v = vae_map[i]
    results[i] = {
        "baseline": classify_only(p),
        "fusion_v1": fuse_v1(p, s, v),
        "fusion_v2": fuse_v2(p, s, v),
        "stream": s,
        "vae_mse": v,
        "p_benign": float(p[BENIGN_IDX]),
        "p_lenc":   float(p[LENC_IDX]),
        "p_ran":    float(p[RAN_IDX]),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def get_label_3way(label):
    if label in RANSOMWARE_CLASSES: return "Ransomware"
    if label in LEGIT_ENC_CLASSES:  return "Legit-Enc"
    if label in BENIGN_CLASSES:     return "Benign"
    return "unknown"


lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(s)

log("=" * 70)
log(f"FUSION V2 — Full-Scale Results ({len(val_idx)} val + {len(variant_idx)} variants)")
log("=" * 70)

# --- Val FPR ---
log("\n── VAL SET FPR (non-ransomware executions misclassified as Attack) ──")
log(f"{'Class':<22} {'Baseline':>12} {'Fusion v1 (OR)':>16} {'Fusion v2 (AND)':>17}")
log("-" * 70)

fpr_results = {}
for cls_name, cls_list in [("Benign", BENIGN_CLASSES), ("Legit-Enc", LEGIT_ENC_CLASSES), ("Ransomware", RANSOMWARE_CLASSES)]:
    cls_idx = [i for i in val_idx if all_labels[i] in cls_list]
    n = len(cls_idx)
    if n == 0: continue
    b  = sum(1 for i in cls_idx if results[i]["baseline"]  == "Attack")
    v1 = sum(1 for i in cls_idx if results[i]["fusion_v1"] == "Attack")
    v2 = sum(1 for i in cls_idx if results[i]["fusion_v2"] == "Attack")
    label = "TPR ↑" if cls_name == "Ransomware" else "FPR ↓"
    log(f"  {cls_name:<20} {b}/{n} ({100*b/n:.0f}%) {label}  {v1}/{n} ({100*v1/n:.0f}%)  {v2}/{n} ({100*v2/n:.0f}%)")
    fpr_results[cls_name] = {"n": n, "baseline": b, "v1": v1, "v2": v2}

# Combined non-ransomware FPR
non_ran_idx = [i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES]
n_nr = len(non_ran_idx)
b_nr  = sum(1 for i in non_ran_idx if results[i]["baseline"]  == "Attack")
v1_nr = sum(1 for i in non_ran_idx if results[i]["fusion_v1"] == "Attack")
v2_nr = sum(1 for i in non_ran_idx if results[i]["fusion_v2"] == "Attack")
log(f"\n  {'Combined non-ransomware FPR':<20} {b_nr}/{n_nr} ({100*b_nr/n_nr:.0f}%)  {v1_nr}/{n_nr} ({100*v1_nr/n_nr:.0f}%)  {v2_nr}/{n_nr} ({100*v2_nr/n_nr:.0f}%)")
log(f"  Local reference (NB10/NB14 @ thr=0.65): combined FPR was 25%")

# --- Variant recall ---
log("\n── VARIANT RECALL (zero-day detection) ──")
log(f"{'Variant':<14} {'N':>4} {'Baseline':>12} {'Fusion v1 (OR)':>16} {'Fusion v2 (AND)':>17}")
log("-" * 68)

variant_results = {}
for vname in VARIANT_CLASSES:
    v_idx = [i for i in variant_idx if all_labels[i] == vname]
    n = len(v_idx)
    if n == 0: continue
    b  = sum(1 for i in v_idx if results[i]["baseline"]  == "Attack")
    v1 = sum(1 for i in v_idx if results[i]["fusion_v1"] == "Attack")
    v2 = sum(1 for i in v_idx if results[i]["fusion_v2"] == "Attack")
    i1 = sum(1 for i in v_idx if results[i]["fusion_v1"] == "Idle")
    i2 = sum(1 for i in v_idx if results[i]["fusion_v2"] == "Idle")
    flag = " ◀" if vname == "Conti_06" else ""
    log(f"  {vname+flag:<14} {n:>4} {b}/{n} ({100*b/n:.0f}%)  {v1}/{n} ({100*v1/n:.0f}%)(+{i1}Idle)  {v2}/{n} ({100*v2/n:.0f}%)(+{i2}Idle)")
    variant_results[vname] = {"n": n, "baseline": b, "v1_attack": v1, "v2_attack": v2, "v1_idle": i1, "v2_idle": i2}

all_b  = sum(1 for i in variant_idx if results[i]["baseline"]  == "Attack")
all_v1 = sum(1 for i in variant_idx if results[i]["fusion_v1"] == "Attack")
all_v2 = sum(1 for i in variant_idx if results[i]["fusion_v2"] == "Attack")
n_var  = len(variant_idx)
log(f"  {'ALL variants':<14} {n_var:>4} {all_b}/{n_var} ({100*all_b/n_var:.0f}%)  {all_v1}/{n_var} ({100*all_v1/n_var:.0f}%)  {all_v2}/{n_var} ({100*all_v2/n_var:.0f}%)")
log(f"  Local reference (NB14 @ thr=0.65): baseline 88.6% → fusion-v1 97.1%")

# --- Verdict ---
log("\n── VERDICT ──")
log(f"  VAE FPR@95 (full-scale): 15.3%  (was 81.9% locally — fix confirmed)")
log(f"  Fusion v1 (OR) combined FPR:  {100*v1_nr/n_nr:.1f}%  (target: <25%)")
log(f"  Fusion v2 (AND) combined FPR: {100*v2_nr/n_nr:.1f}%  (target: <25%)")
log(f"  Fusion v2 Conti_06 recall: {100*variant_results['Conti_06']['v2_attack']/variant_results['Conti_06']['n']:.0f}%  (target: ≥80%)")

if v2_nr/n_nr < 0.25 and variant_results["Conti_06"]["v2_attack"]/variant_results["Conti_06"]["n"] >= 0.80:
    log(f"\n  ✅ Fusion v2 (AND) SUCCEEDS: FPR controlled + Conti_06 recall maintained")
elif v1_nr/n_nr < 0.25:
    log(f"\n  ✅ Fusion v1 (OR) hits FPR target. v2 AND rule: check Conti_06 recall.")
else:
    log(f"\n  ⚠  FPR still above 25% — check threshold or VAE signal strength")

# --- Mix set (simultaneous legit-enc + ransomware) ---
if len(mix_idx) > 0:
    log("\n── MIX SET (simultaneous legit-enc + ransomware) ──")
    mix_classes = sorted(set(all_labels[i] for i in mix_idx))
    for mc in mix_classes[:5]:   # show first 5
        mc_idx = [i for i in mix_idx if all_labels[i] == mc]
        n = len(mc_idx)
        v2 = sum(1 for i in mc_idx if results[i]["fusion_v2"] == "Attack")
        log(f"  {mc:<30} {v2}/{n} ({100*v2/n:.0f}%) detected as Attack")

log("\n" + "=" * 70)

# Save
summary_text = "\n".join(lines)

with open(FEATURES_DIR / "fusion_v2_results.txt", "w") as f:
    f.write(summary_text)

json_out = {
    "thresholds": {"stream": STREAM_THRESH, "vae": vae_threshold, "W_stream": W_STREAM},
    "val_fpr": fpr_results,
    "combined_nonran_fpr": {"n": n_nr, "baseline": b_nr, "v1": v1_nr, "v2": v2_nr},
    "variant_recall": variant_results,
    "local_reference": {
        "fpr_local": 0.25, "conti06_local_v1": 0.80, "variant_recall_local_v1": 0.971
    }
}
with open(FEATURES_DIR / "fusion_v2_results.json", "w") as f:
    json.dump(json_out, f, indent=2)

print(f"\nSaved:")
print(f"  {FEATURES_DIR}/fusion_v2_results.txt")
print(f"  {FEATURES_DIR}/fusion_v2_results.json")
print("Done.")
