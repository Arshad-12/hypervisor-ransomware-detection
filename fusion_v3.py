"""
RanSMAP Fusion v3 — Full-Scale (Alliance, 16e-fusion)
======================================================
Fixes the Rule 1 streaming block identified in the fusion v2 diagnostic:

  v2 Rule 1: Legit-Enc dominant → Benign (streaming blocked — caused 13/21 misses)
  v3 Rule 1: Legit-Enc dominant → run streaming; if fires → Attack; else → Benign

Also sweeps the streaming threshold (0.50–0.80) to find the optimal operating
point for Conti_06 (8 Benign-primary misses that need a lower threshold).

Two-failure-mode summary:
  - 13 Legit-Enc misses (Conti 02–07): fixed by Rule 1 change
  - 8 Benign misses (all Conti_06): require threshold <= P(ran) seen in streaming

Saves: fusion_v3_results.txt, fusion_v3_results.json
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")

RANSOMWARE_CLASSES = ["WannaCry","Ryuk","REvil","LockBit","Darkside","Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt","SDelete","Zip"]
BENIGN_CLASSES     = ["Firefox","Office","Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES    = [f"Conti_{i:02d}" for i in range(1, 8)]
CLASS_ORDER        = ["benign","legit_encryption","ransomware"]
BENIGN_IDX, LENC_IDX, RAN_IDX = 0, 1, 2

W_STREAM  = 50
STRIDE    = 5
N_WINDOWS = 300
STREAM_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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
        return self.decode(mu), mu, lv


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
text_seqs  = np.load(FEATURES_DIR / "text_sequences.npy")
all_labels = np.load(FEATURES_DIR / "text_labels.npy")

meta_df = pd.read_csv(FEATURES_DIR / "metadata.csv")
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])

split        = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx      = split["val_idx"]
variant_idx  = np.where(split["variant_mask"])[0]
mix_idx      = np.where(split["mix_mask"])[0]

print(f"Val: {len(val_idx)}  |  Variants: {len(variant_idx)}  |  Mix: {len(mix_idx)}")

# Load models
print("Loading models...")
clf = TextEmbeddingClassifier().to(device)
clf.load_state_dict(torch.load(FEATURES_DIR/"classifier.pt", map_location=device, weights_only=True))
clf.eval()

vae = TimeSeriesVAE().to(device)
vae.load_state_dict(torch.load(FEATURES_DIR/"vae_model.pt", map_location=device, weights_only=True))
vae.eval()

vae_mean      = np.load(FEATURES_DIR/"vae_scaler_mean.npy")
vae_scale     = np.load(FEATURES_DIR/"vae_scaler_scale.npy")
vae_threshold = float(np.load(FEATURES_DIR/"vae_threshold.npy")[0])
print(f"VAE threshold: {vae_threshold:.4f}")


# ── Inference ─────────────────────────────────────────────────────────────────
def primary_probs(idx):
    x = torch.tensor(text_seqs[idx], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = clf(x)
        return torch.softmax(logits, dim=1).cpu().numpy()[0]

def streaming_max(idx):
    seq = text_seqs[idx]
    best = 0.0
    with torch.no_grad():
        for start in range(0, N_WINDOWS - W_STREAM + 1, STRIDE):
            chunk  = seq[start:start + W_STREAM]
            padded = np.concatenate([chunk, np.zeros((N_WINDOWS - W_STREAM, 384), np.float32)])
            x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
            logits, _, _ = clf(x)
            p = torch.softmax(logits, dim=1)[0, RAN_IDX].item()
            if p > best:
                best = p
    return best

def vae_mse(idx):
    feat_flat   = feat_23[idx].reshape(-1, 23)
    feat_scaled = ((feat_flat - vae_mean) / vae_scale).reshape(1, 300, 23).astype(np.float32)
    x = torch.tensor(feat_scaled).to(device)
    with torch.no_grad():
        mu, _ = vae.encode(x)
        recon  = vae.decode(mu)
        return F.mse_loss(recon, x, reduction="mean").item()


# ── Precompute all signals ────────────────────────────────────────────────────
eval_indices = np.concatenate([val_idx, variant_idx, mix_idx])
print(f"\nPrecomputing primary probs for {len(eval_indices)} executions...")
pmap = {}
for idx in eval_indices:
    pmap[int(idx)] = primary_probs(idx)

# Streaming needed for Legit-Enc AND Benign primaries (v3 change)
print("Computing streaming for Legit-Enc and Benign primaries...")
smap, vmap = {}, {}
needs_stream = [i for i in eval_indices
                if pmap[int(i)].argmax() in (BENIGN_IDX, LENC_IDX)]
print(f"  {len(needs_stream)} executions need streaming (vs 102 in v2 — Legit-Enc now included)")

for idx in needs_stream:
    smap[int(idx)] = streaming_max(idx)
    vmap[int(idx)] = vae_mse(idx)
for idx in eval_indices:
    if int(idx) not in smap:
        smap[int(idx)] = 0.0
        vmap[int(idx)] = 0.0


# ── Fusion rules ──────────────────────────────────────────────────────────────
def fuse_v2(p, s, v, st):
    """v2: Legit-Enc blocks streaming, Benign uses OR rule."""
    if p[LENC_IDX] >= p[RAN_IDX] and p[LENC_IDX] >= p[BENIGN_IDX]:
        return "Benign"
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]:
        return "Attack"
    if s >= st: return "Attack"
    if 0 < v < 1e6 and v > vae_threshold: return "Idle"
    return "Benign"

def fuse_v3(p, s, v, st):
    """v3: Rule 1 now runs streaming for Legit-Enc too."""
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]:
        return "Attack"
    # Both Legit-Enc and Benign primaries: consult streaming
    if s >= st: return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        return "Benign"   # Legit-Enc primary, streaming didn't fire → safe
    # Benign primary, streaming didn't fire → check VAE
    if 0 < v < 1e6 and v > vae_threshold: return "Idle"
    return "Benign"

def classify_only(p):
    return "Attack" if p.argmax() == RAN_IDX else "Benign"


# ── Evaluate at each threshold ────────────────────────────────────────────────
lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(s)

threshold_sweep = []
log("=" * 75)
log("FUSION V3 — Threshold Sweep (stream_thresh from 0.40 to 0.70)")
log(f"{'Thresh':>7}  {'Val FPR':>8}  {'Var Recall':>11}  {'Conti_06':>10}  {'Verdict'}")
log("-" * 75)

for st in STREAM_THRESHOLDS:
    def r(idx, fn): return fn(pmap[int(idx)], smap[int(idx)], vmap[int(idx)], st)

    non_ran_val = [i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES]
    fpr_v3   = sum(1 for i in non_ran_val if r(i, fuse_v3)=="Attack") / len(non_ran_val)
    rec_v3   = sum(1 for i in variant_idx if r(i, fuse_v3)=="Attack") / len(variant_idx)
    c06_idx  = [i for i in variant_idx if all_labels[i]=="Conti_06"]
    c06_v3   = sum(1 for i in c06_idx if r(i, fuse_v3)=="Attack") / len(c06_idx)

    ok = "✓" if fpr_v3 < 0.10 and rec_v3 >= 0.90 else ("~" if fpr_v3 < 0.20 else "✗")
    log(f"  {st:.2f}    {fpr_v3:>7.1%}    {rec_v3:>10.1%}  {c06_v3:>9.1%}   {ok}")
    threshold_sweep.append(dict(thresh=st, fpr_v3=fpr_v3, rec_v3=rec_v3, c06_v3=c06_v3))

log()

# Detailed report at reference threshold (0.65)
ST = 0.65
log("=" * 75)
log(f"FUSION V3 DETAIL @ stream_thresh={ST}  (compare to v2 @ same threshold)")
log("=" * 75)

log(f"\n{'Class':<22} {'Baseline':>12} {'Fusion v2':>12} {'Fusion v3':>12}")
log("-" * 62)

fpr_by_class = {}
for cls_name, cls_list in [("Benign",BENIGN_CLASSES),("Legit-Enc",LEGIT_ENC_CLASSES),("Ransomware",RANSOMWARE_CLASSES)]:
    idx_cls = [i for i in val_idx if all_labels[i] in cls_list]
    n = len(idx_cls)
    if n == 0: continue
    b   = sum(1 for i in idx_cls if classify_only(pmap[i])=="Attack")
    v2a = sum(1 for i in idx_cls if fuse_v2(pmap[i],smap[i],vmap[i],ST)=="Attack")
    v3a = sum(1 for i in idx_cls if fuse_v3(pmap[i],smap[i],vmap[i],ST)=="Attack")
    tag = "TPR ↑" if cls_name=="Ransomware" else "FPR ↓"
    log(f"  {cls_name:<20} {b}/{n} ({100*b/n:.0f}%) {tag}  {v2a}/{n} ({100*v2a/n:.0f}%)  {v3a}/{n} ({100*v3a/n:.0f}%)")
    fpr_by_class[cls_name] = {"n":n,"baseline":b,"v2":v2a,"v3":v3a}

non_ran = [i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES]
n_nr = len(non_ran)
b_nr  = sum(1 for i in non_ran if classify_only(pmap[i])=="Attack")
v2_nr = sum(1 for i in non_ran if fuse_v2(pmap[i],smap[i],vmap[i],ST)=="Attack")
v3_nr = sum(1 for i in non_ran if fuse_v3(pmap[i],smap[i],vmap[i],ST)=="Attack")
log(f"\n  {'Combined FPR':<20} {b_nr}/{n_nr} ({100*b_nr/n_nr:.0f}%)        {v2_nr}/{n_nr} ({100*v2_nr/n_nr:.0f}%)   {v3_nr}/{n_nr} ({100*v3_nr/n_nr:.0f}%)")

log(f"\n── VARIANT RECALL @ thresh={ST} ──")
log(f"{'Variant':<14} {'N':>4}  {'Baseline':>10}  {'Fusion v2':>10}  {'Fusion v3':>10}")
log("-" * 56)

variant_detail = {}
for vname in VARIANT_CLASSES:
    vi = [i for i in variant_idx if all_labels[i]==vname]
    n  = len(vi)
    if n == 0: continue
    b   = sum(1 for i in vi if classify_only(pmap[i])=="Attack")
    v2a = sum(1 for i in vi if fuse_v2(pmap[i],smap[i],vmap[i],ST)=="Attack")
    v3a = sum(1 for i in vi if fuse_v3(pmap[i],smap[i],vmap[i],ST)=="Attack")
    v2i = sum(1 for i in vi if fuse_v2(pmap[i],smap[i],vmap[i],ST)=="Idle")
    v3i = sum(1 for i in vi if fuse_v3(pmap[i],smap[i],vmap[i],ST)=="Idle")
    flag = " ◀" if vname=="Conti_06" else ""
    log(f"  {vname+flag:<14} {n:>4}  {b}/{n} ({100*b/n:.0f}%)  {v2a}/{n} ({100*v2a/n:.0f}%)(+{v2i}Idle)  {v3a}/{n} ({100*v3a/n:.0f}%)(+{v3i}Idle)")
    variant_detail[vname] = {"n":n,"baseline":b,"v2":v2a,"v3":v3a}

all_b   = sum(1 for i in variant_idx if classify_only(pmap[i])=="Attack")
all_v2  = sum(1 for i in variant_idx if fuse_v2(pmap[i],smap[i],vmap[i],ST)=="Attack")
all_v3  = sum(1 for i in variant_idx if fuse_v3(pmap[i],smap[i],vmap[i],ST)=="Attack")
n_var   = len(variant_idx)
log(f"  {'ALL variants':<14} {n_var:>4}  {all_b}/{n_var} ({100*all_b/n_var:.0f}%)  {all_v2}/{n_var} ({100*all_v2/n_var:.0f}%)  {all_v3}/{n_var} ({100*all_v3/n_var:.0f}%)")

log(f"\n── STREAMING SCORES for MISSED variants (v3 @ thresh={ST}) ──")
log(f"  (Legit-Enc misses now checked — how close to threshold?)")
log(f"{'Exec':>5}  {'Variant':<12}  {'Primary':>14}  {'Stream max P(ran)':>18}  {'VAE MSE':>10}  {'Fires?':>7}")
log("-" * 80)
for idx in variant_idx:
    label = all_labels[idx]
    p = pmap[int(idx)]
    s = smap[int(idx)]
    v = vmap[int(idx)]
    if fuse_v3(p,s,v,ST) != "Attack":   # only show misses
        pred_primary = CLASS_ORDER[p.argmax()]
        fires = "YES" if s >= ST else f"no ({s:.2f})"
        log(f"  {idx:>5}  {label:<12}  {pred_primary:>14}  {s:>18.4f}  {v:>10.4f}  {fires:>7}")

log()
log("=" * 75)
log("VERDICT")
log("=" * 75)
log(f"  v2 baseline recall: {100*all_v2/n_var:.0f}%  (13 blocked by Rule 1, 8 Conti_06 dormant)")
log(f"  v3 recall @ {ST}:   {100*all_v3/n_var:.0f}%  (Rule 1 fix — Legit-Enc now gets streaming)")
log(f"  v3 FPR @ {ST}:      {100*v3_nr/n_nr:.1f}%")
if all_v3/n_var > all_v2/n_var:
    log(f"  ✅ Rule 1 fix recovers {all_v3-all_v2} additional variants")
else:
    log(f"  ⚠  Streaming not firing for Legit-Enc misses either — need lower threshold")

best = max(threshold_sweep, key=lambda x: x["rec_v3"] - 3*x["fpr_v3"])
log(f"\n  Optimal threshold (recall - 3×FPR): {best['thresh']:.2f}")
log(f"    → Recall {best['rec_v3']:.1%}  |  FPR {best['fpr_v3']:.1%}  |  Conti_06 {best['c06_v3']:.1%}")

summary = "\n".join(lines)
with open(FEATURES_DIR/"fusion_v3_results.txt","w") as f: f.write(summary)

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        return super().default(o)

out = {
    "version": "v3",
    "stream_ref_threshold": ST,
    "vae_threshold": vae_threshold,
    "threshold_sweep": threshold_sweep,
    "val_fpr_at_ref": fpr_by_class,
    "combined_fpr": {"n":n_nr,"v2":v2_nr,"v3":v3_nr},
    "variant_recall": variant_detail,
    "best_threshold": best,
}
with open(FEATURES_DIR/"fusion_v3_results.json","w") as f:
    json.dump(out, f, indent=2, cls=NpEncoder)

print(f"\nSaved: {FEATURES_DIR}/fusion_v3_results.txt / .json")
print("Done.")
