"""
RanSMAP Fusion v4 — Full-Scale (Alliance)
==========================================
Key finding from v3 diagnostic:
  - 13 Legit-Enc misses: streaming flat-zero (0.00–0.16) — can't help
  - BUT VAE MSE > threshold for 10/13 of those same misses
  - The VAE knows what real AESCrypt/Zip/SDelete looks like (it was trained on them)
    and correctly flags Conti variants that LOOK like legit-enc as anomalous

v4 change: extend VAE check to Legit-Enc primary path
  v3 Rule 1 (Legit-Enc): if stream fires → Attack; else → Benign
  v4 Rule 1 (Legit-Enc): if stream fires OR VAE fires → Attack; else → Benign

Expected: 10/13 Legit-Enc misses recovered → recall 70% → ~84%

Saves: fusion_v4_results.txt, fusion_v4_results.json
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")

RANSOMWARE_CLASSES = ["WannaCry","Ryuk","REvil","LockBit","Darkside","Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt","SDelete","Zip"]
BENIGN_CLASSES     = ["Firefox","Office","Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES    = [f"Conti_{i:02d}" for i in range(1, 8)]
CLASS_ORDER        = ["benign","legit_encryption","ransomware"]
BENIGN_IDX, LENC_IDX, RAN_IDX = 0, 1, 2

STREAM_THRESH = 0.65
W_STREAM      = 50
STRIDE        = 5
N_WINDOWS     = 300

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
text_seqs  = np.load(FEATURES_DIR/"text_sequences.npy")
all_labels = np.load(FEATURES_DIR/"text_labels.npy")

meta_df = pd.read_csv(FEATURES_DIR/"metadata.csv")
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])

split       = np.load(FEATURES_DIR/"fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx     = np.where(split["mix_mask"])[0]

print(f"Val: {len(val_idx)}  |  Variants: {len(variant_idx)}  |  Mix: {len(mix_idx)}")

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
            if p > best: best = p
    return best

def vae_mse(idx):
    feat_flat   = feat_23[idx].reshape(-1, 23)
    feat_scaled = ((feat_flat - vae_mean) / vae_scale).reshape(1, 300, 23).astype(np.float32)
    x = torch.tensor(feat_scaled).to(device)
    with torch.no_grad():
        mu, _ = vae.encode(x)
        recon  = vae.decode(mu)
        return F.mse_loss(recon, x, reduction="mean").item()


# ── Precompute signals ────────────────────────────────────────────────────────
eval_indices = np.concatenate([val_idx, variant_idx, mix_idx])
print(f"\nPrecomputing primary probs for {len(eval_indices)} executions...")
pmap = {int(idx): primary_probs(idx) for idx in eval_indices}

# Streaming for Benign + Legit-Enc primaries; VAE for all non-Ransomware primaries
print("Computing streaming + VAE for non-Ransomware primaries...")
smap, vmap = {}, {}
for idx in eval_indices:
    i = int(idx)
    if pmap[i].argmax() != RAN_IDX:
        smap[i] = streaming_max(idx)
        vmap[i] = vae_mse(idx)
    else:
        smap[i] = 0.0
        vmap[i] = 0.0

lenc_checked = sum(1 for i in eval_indices if pmap[int(i)].argmax() == LENC_IDX)
ben_checked  = sum(1 for i in eval_indices if pmap[int(i)].argmax() == BENIGN_IDX)
print(f"  Legit-Enc primaries: {lenc_checked}  |  Benign primaries: {ben_checked}")


# ── Fusion rules ──────────────────────────────────────────────────────────────
def vae_fires(v):
    return 0 < v < 1e6 and v > vae_threshold

def fuse_baseline(p):
    return "Attack" if p.argmax() == RAN_IDX else "Benign"

def fuse_v2(p, s, v):
    """v2: Legit-Enc blocks everything; Benign → streaming only."""
    if p[LENC_IDX] >= p[RAN_IDX] and p[LENC_IDX] >= p[BENIGN_IDX]: return "Benign"
    if p[RAN_IDX]  >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]: return "Attack"
    if s >= STREAM_THRESH: return "Attack"
    if vae_fires(v): return "Idle"
    return "Benign"

def fuse_v4(p, s, v):
    """v4: VAE added to Legit-Enc path; Benign → streaming OR VAE."""
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]:
        return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        # Legit-Enc primary: check BOTH streaming and VAE
        if s >= STREAM_THRESH or vae_fires(v): return "Attack"
        return "Benign"
    # Benign primary: streaming first, then VAE
    if s >= STREAM_THRESH: return "Attack"
    if vae_fires(v): return "Idle"
    return "Benign"


# ── Evaluate ──────────────────────────────────────────────────────────────────
lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(s)

log("=" * 72)
log("FUSION V4 — VAE extended to Legit-Enc primary path")
log(f"  VAE threshold: {vae_threshold:.4f}  |  Stream threshold: {STREAM_THRESH}")
log("=" * 72)

# Val FPR
log(f"\n── VAL SET FPR ──")
log(f"{'Class':<22} {'Baseline':>12} {'Fusion v2':>12} {'Fusion v4':>12}")
log("-" * 62)

fpr_out = {}
for cls_name, cls_list in [("Benign",BENIGN_CLASSES),("Legit-Enc",LEGIT_ENC_CLASSES),("Ransomware",RANSOMWARE_CLASSES)]:
    idx_c = [i for i in val_idx if all_labels[i] in cls_list]
    n = len(idx_c)
    if n == 0: continue
    b   = sum(1 for i in idx_c if fuse_baseline(pmap[i])=="Attack")
    v2a = sum(1 for i in idx_c if fuse_v2(pmap[i],smap[i],vmap[i])=="Attack")
    v4a = sum(1 for i in idx_c if fuse_v4(pmap[i],smap[i],vmap[i])=="Attack")
    tag = "TPR ↑" if cls_name=="Ransomware" else "FPR ↓"
    log(f"  {cls_name:<20} {b}/{n} ({100*b/n:.0f}%) {tag}  {v2a}/{n} ({100*v2a/n:.0f}%)  {v4a}/{n} ({100*v4a/n:.0f}%)")
    fpr_out[cls_name] = {"n":n,"baseline":b,"v2":v2a,"v4":v4a}

non_ran = [i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES]
n_nr = len(non_ran)
b_nr  = sum(1 for i in non_ran if fuse_baseline(pmap[i])=="Attack")
v2_nr = sum(1 for i in non_ran if fuse_v2(pmap[i],smap[i],vmap[i])=="Attack")
v4_nr = sum(1 for i in non_ran if fuse_v4(pmap[i],smap[i],vmap[i])=="Attack")
log(f"\n  {'Combined FPR':<20} {b_nr}/{n_nr} ({100*b_nr/n_nr:.0f}%)        {v2_nr}/{n_nr} ({100*v2_nr/n_nr:.0f}%)   {v4_nr}/{n_nr} ({100*v4_nr/n_nr:.0f}%)")

# Variant recall
log(f"\n── VARIANT RECALL ──")
log(f"{'Variant':<14} {'N':>4}  {'Baseline':>10}  {'Fusion v2':>10}  {'Fusion v4':>10}  {'VAE MSE':>10}")
log("-" * 68)

var_out = {}
for vname in VARIANT_CLASSES:
    vi  = [i for i in variant_idx if all_labels[i]==vname]
    n   = len(vi)
    if n == 0: continue
    b   = sum(1 for i in vi if fuse_baseline(pmap[i])=="Attack")
    v2a = sum(1 for i in vi if fuse_v2(pmap[i],smap[i],vmap[i])=="Attack")
    v4a = sum(1 for i in vi if fuse_v4(pmap[i],smap[i],vmap[i])=="Attack")
    v4i = sum(1 for i in vi if fuse_v4(pmap[i],smap[i],vmap[i])=="Idle")
    avg_vae = np.mean([vmap[i] for i in vi])
    flag = " ◀" if vname=="Conti_06" else ""
    log(f"  {vname+flag:<14} {n:>4}  {b}/{n} ({100*b/n:.0f}%)  {v2a}/{n} ({100*v2a/n:.0f}%)  {v4a}/{n} ({100*v4a/n:.0f}%)(+{v4i}Idle)  {avg_vae:>10.4f}")
    var_out[vname] = {"n":n,"baseline":b,"v2":v2a,"v4":v4a,"v4_idle":v4i}

all_b  = sum(1 for i in variant_idx if fuse_baseline(pmap[i])=="Attack")
all_v2 = sum(1 for i in variant_idx if fuse_v2(pmap[i],smap[i],vmap[i])=="Attack")
all_v4 = sum(1 for i in variant_idx if fuse_v4(pmap[i],smap[i],vmap[i])=="Attack")
n_var  = len(variant_idx)
log(f"  {'ALL':14} {n_var:>4}  {all_b}/{n_var} ({100*all_b/n_var:.0f}%)  {all_v2}/{n_var} ({100*all_v2/n_var:.0f}%)  {all_v4}/{n_var} ({100*all_v4/n_var:.0f}%)")

# VAE MSE breakdown for val Legit-Enc (to check FPR impact)
log(f"\n── VAE MSE on val Legit-Enc (FPR check — should stay low) ──")
lenc_val = [i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES]
lenc_mse = [vmap[i] for i in lenc_val]
lenc_above = sum(1 for v in lenc_mse if v > vae_threshold)
log(f"  Val Legit-Enc  n={len(lenc_val)}  mean MSE={np.mean(lenc_mse):.4f}  above threshold={lenc_above} ({100*lenc_above/len(lenc_val):.0f}%)")

ben_val = [i for i in val_idx if all_labels[i] in BENIGN_CLASSES]
ben_mse = [vmap[i] for i in ben_val]
ben_above = sum(1 for v in ben_mse if v > vae_threshold)
log(f"  Val Benign     n={len(ben_val)}  mean MSE={np.mean(ben_mse):.4f}  above threshold={ben_above} ({100*ben_above/len(ben_val):.0f}%)")

# Mix set
if len(mix_idx) > 0:
    log(f"\n── MIX SET (simultaneous legit-enc + ransomware) ──")
    for mc in sorted(set(all_labels[i] for i in mix_idx))[:6]:
        mc_idx = [i for i in mix_idx if all_labels[i]==mc]
        v4a = sum(1 for i in mc_idx if fuse_v4(pmap[i],smap[i],vmap[i])=="Attack")
        log(f"  {mc:<30} {v4a}/{len(mc_idx)} ({100*v4a/len(mc_idx):.0f}%) detected")

# Verdict
log(f"\n── VERDICT ──")
log(f"  Baseline recall:    {100*all_b/n_var:.0f}%  (classifier alone)")
log(f"  Fusion v2 recall:   {100*all_v2/n_var:.0f}%  (streaming OR VAE, Legit-Enc blocked)")
log(f"  Fusion v4 recall:   {100*all_v4/n_var:.0f}%  (VAE added to Legit-Enc path)")
log(f"  Combined FPR v4:    {100*v4_nr/n_nr:.1f}%")
log(f"  Conti_06 recall:    {100*var_out['Conti_06']['v4']/var_out['Conti_06']['n']:.0f}%")
log()
recovered = all_v4 - all_v2
if recovered > 0:
    log(f"  ✅ VAE in Legit-Enc path recovered {recovered} additional variants")
    log(f"     ({100*all_v4/n_var:.0f}% vs {100*all_v2/n_var:.0f}% with fusion v2)")
if v4_nr/n_nr < 0.10:
    log(f"  ✅ FPR controlled at {100*v4_nr/n_nr:.1f}%")
elif v4_nr/n_nr < 0.20:
    log(f"  ⚠  FPR at {100*v4_nr/n_nr:.1f}% — acceptable but check Legit-Enc val MSE above")
else:
    log(f"  ✗  FPR {100*v4_nr/n_nr:.1f}% — VAE fires too broadly on val Legit-Enc")

summary = "\n".join(lines)
with open(FEATURES_DIR/"fusion_v4_results.txt","w") as f: f.write(summary)

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o,(np.integer,np.floating,np.bool_)): return o.item()
        return super().default(o)

out = {
    "version": "v4",
    "stream_thresh": STREAM_THRESH,
    "vae_threshold": vae_threshold,
    "val_fpr": fpr_out,
    "combined_fpr": {"n":n_nr,"baseline":b_nr,"v2":v2_nr,"v4":v4_nr},
    "variant_recall": var_out,
    "val_lenc_vae": {"n":len(lenc_val),"mean_mse":float(np.mean(lenc_mse)),"above_thresh":lenc_above},
    "val_benign_vae": {"n":len(ben_val),"mean_mse":float(np.mean(ben_mse)),"above_thresh":ben_above},
}
with open(FEATURES_DIR/"fusion_v4_results.json","w") as f:
    json.dump(out, f, indent=2, cls=NpEncoder)

print(f"\nSaved: {FEATURES_DIR}/fusion_v4_results.txt / .json")
print("Done.")
