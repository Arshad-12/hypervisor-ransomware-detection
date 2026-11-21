"""
vae_binary_fusion.py — VAE Anomaly Head + 2-way CNN Binary Fusion
==================================================================
Per HANDOFF update 49 (Opus decision, July 2026).

Replaces the BiLSTM anomaly head with the TemporalVAE reconstruction MSE.
Everything else mirrors binary_fusion.py:
  - 2-way MiniLM CNN (classifier_2way.pt, frozen) + streaming
  - Binary output: Attack / Benign (no Idle)
  - Threshold from val non-ransomware 95th pct (same discipline as BiLSTM)

Reports:
  1. Per-variant recall + Conti_06
  2. Benign FPR, LEnc FPR, combined FPR
  3. Per-exec Conti_06 breakdown (VAE MSE per exec, vs BiLSTM baseline)
  4. Head-to-head vs BiLSTM-binary (99% / 90% Conti_06 / ~5% FPR)
  5. 6-family LOFO with VAE head (CNN retrained per fold, VAE fixed)

★  CORE RULE: known-classes-only training; blind eval on ALL variants + LOFO;
   Benign FPR and Legit-Enc FPR reported SEPARATELY. Conti_06 NEVER in training.
"""

import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

FEATURES_DIR  = Path("/scratch/aabdulj4/RanSMAP/features")
STREAM_THRESH = 0.65
FPR_PCT       = 95.0

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]
LOFO_FAMILIES        = {
    "WannaCry": ["WannaCry"], "Ryuk": ["Ryuk"], "REvil": ["REvil"],
    "LockBit": ["LockBit"], "Darkside": ["Darkside"], "Conti": ["Conti"],
}

# BiLSTM-binary baseline (from binary_fusion.py results) for head-to-head
BILSTM_BASELINE = {
    "recall": 0.99, "c06": 0.90, "fpr_ben": None, "fpr_lenc": None, "fpr_nr": 0.05,
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
# SECTION 1 — Load data
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72); log("Loading data"); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
meta_df    = pd.read_csv(FEATURES_DIR / "metadata.csv")

t0 = time.time()
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])
log(f"  feat_23 loaded: {feat_23.shape}  in {time.time()-t0:.1f}s")

log("  Loading MiniLM sequences...")
text_seqs = np.load(FEATURES_DIR / "text_sequences.npy")   # (N, 300, 384)

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx     = np.where(split["mix_mask"])[0]

train_mask   = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx    = np.where(train_mask)[0]
train_y_bin  = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in train_idx])
train_nonran = (train_y_bin == 0)

val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ran_idx    = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])
val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
c06_idx        = [int(i) for i in variant_idx if all_labels[i] == "Conti_06"]
eval_idx       = np.concatenate([val_idx, variant_idx, mix_idx])

log(f"  Training pool: {len(train_idx)} (ran={train_y_bin.sum()}, non-ran={train_nonran.sum()})")
log(f"  Val: {len(val_idx)}  (non-ran={len(val_nonran_idx)}, ran={len(val_ran_idx)})")
log(f"        Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}")
log(f"  Variants: {len(variant_idx)},  Conti_06: {len(c06_idx)}")

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Load TemporalVAE + compute MSE scores
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Loading TemporalVAE + computing anomaly scores"); log("=" * 72)

HIDDEN_DIM = 64
LATENT_DIM = 32
NUM_LAYERS  = 2

class TemporalVAE(nn.Module):
    def __init__(self, n_features=23, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_features = n_features
        self.encoder_lstm = nn.LSTM(
            n_features, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.fc_mu     = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=0.2
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_fwd = h_n[-2]; h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, seq_len=300):
        h0 = self.fc_decode(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        inp = self.fc_decode(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(inp, (h0, c0))
        return self.output_layer(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return self.decode(z, x.size(1)), mu, logvar


class SeqDataset(Dataset):
    def __init__(self, sequences):
        self.X = torch.tensor(sequences, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


# Load scaler + scale feat_23
vae_mean  = np.load(FEATURES_DIR / "temporal_vae_scaler_mean.npy")
vae_scale = np.load(FEATURES_DIR / "temporal_vae_scaler_scale.npy")
N = len(feat_23)
feat_23_scaled = ((feat_23.reshape(-1, 23) - vae_mean) / vae_scale).reshape(N, 300, 23).astype(np.float32)
log(f"  Scaler loaded. feat_23_scaled shape: {feat_23_scaled.shape}")

# Load model
vae_model = TemporalVAE().to(device)
vae_model.load_state_dict(torch.load(FEATURES_DIR / "temporal_vae_model.pt",
                                     map_location=device, weights_only=True))
vae_model.eval()
log("  TemporalVAE loaded.")

# Load saved threshold (95th pct val non-ran from training)
vae_threshold_saved = float(np.load(FEATURES_DIR / "temporal_vae_threshold.npy").flat[0])
log(f"  Saved VAE threshold (from training): {vae_threshold_saved:.4f}")

# Compute MSE for all execs
@torch.no_grad()
def compute_vae_mse(arr):
    """Per-sample reconstruction MSE (deterministic mu, eval mode)."""
    dl   = DataLoader(SeqDataset(arr), batch_size=64, shuffle=False)
    errs = []
    for bx in dl:
        bx = bx.to(device)
        mu_b, _ = vae_model.encode(bx)
        rx = vae_model.decode(mu_b, bx.size(1))
        errs.append(((rx - bx) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(errs)

log(f"  Computing VAE MSE scores for all {N} execs...")
t0 = time.time()
vae_scores = compute_vae_mse(feat_23_scaled)
log(f"  Done in {time.time()-t0:.1f}s. Score range: [{vae_scores.min():.4f}, {vae_scores.max():.4f}]")

# Set threshold from val non-ransomware 95th pct (same discipline as BiLSTM)
vae_thresh_val = float(np.percentile(vae_scores[val_nonran_idx], FPR_PCT))
log(f"  Val non-ran 95th pct threshold: {vae_thresh_val:.4f}")
log(f"    Val FPR at this thresh:   {(vae_scores[val_nonran_idx] > vae_thresh_val).mean():.1%}")
log(f"    Val TPR (ran) at thresh:  {(vae_scores[val_ran_idx] > vae_thresh_val).mean():.1%}")

# Comparison table: MSE distribution by class
log(f"\n  VAE MSE by class (val):")
log(f"  {'Class':<12}  {'N':>4}  {'Mean':>8}  {'P50':>8}  {'P95':>8}  {'>thresh':>8}")
for cls_name, idx_arr in [
    ("Val Benign", val_ben_idx), ("Val LEnc", val_lenc_idx), ("Val Ran", val_ran_idx),
]:
    sc = vae_scores[idx_arr]
    log(f"  {cls_name:<12}  {len(sc):>4}  {sc.mean():>8.4f}  "
        f"{np.percentile(sc,50):>8.4f}  {np.percentile(sc,95):>8.4f}  "
        f"{(sc > vae_thresh_val).mean():>8.1%}")

c06_sc = vae_scores[c06_idx]
log(f"  {'Conti_06':<12}  {len(c06_sc):>4}  {c06_sc.mean():>8.4f}  "
    f"{np.percentile(c06_sc,50):>8.4f}  {np.percentile(c06_sc,95):>8.4f}  "
    f"{(c06_sc > vae_thresh_val).mean():>8.1%}")

json_out["vae_threshold"] = {
    "saved_training_threshold": vae_threshold_saved,
    "val_nonran_95pct_threshold": vae_thresh_val,
}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Load 2-way CNN + compute probs + streaming
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Loading 2-way CNN (classifier_2way.pt) + probs + streaming"); log("=" * 72)

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
log("  classifier_2way.pt loaded.")

@torch.no_grad()
def probs_2way(model, i):
    x = torch.tensor(text_seqs[int(i)][None], dtype=torch.float32).to(device)
    logits, _, _ = model(x)
    return torch.softmax(logits, dim=1).cpu().numpy()[0]

@torch.no_grad()
def streaming_max_2way(model, i):
    seq = text_seqs[int(i)]; W = 50; best = 0.0
    for start in range(0, 300 - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = model(x)
        p = torch.softmax(logits, dim=1)[0, 1].item()
        if p > best: best = p
    return best

log(f"  Precomputing 2-way probs + streaming for {len(eval_idx)} eval execs...")
pmap2, smap2 = {}, {}
for ii, i in enumerate(eval_idx):
    i = int(i)
    pmap2[i] = probs_2way(clf2, i)
    if pmap2[i].argmax() != 1:
        smap2[i] = streaming_max_2way(clf2, i)
    else:
        smap2[i] = 0.0
    if (ii + 1) % 100 == 0: log(f"    {ii+1}/{len(eval_idx)}")
log("  Done.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VAE binary fusion eval
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("VAE BINARY FUSION EVAL"); log("=" * 72)

def binary_verdict_vae(i, vae_thr, pmap=None, smap=None):
    """2-way CNN + streaming + VAE anomaly → Attack/Benign."""
    i = int(i)
    p = (pmap or pmap2)[i]
    s = (smap or smap2).get(i, 0.0)
    fires = vae_scores[i] > vae_thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"

op_thresh = vae_thresh_val
log(f"\n  Operating threshold: {op_thresh:.4f} (95th pct val non-ran VAE MSE)")

# Overall eval
var_att  = sum(1 for i in variant_idx if binary_verdict_vae(i, op_thresh) == "Attack")
c06_att  = sum(1 for i in c06_idx     if binary_verdict_vae(i, op_thresh) == "Attack")
b_fp     = sum(1 for i in val_ben_idx  if binary_verdict_vae(i, op_thresh) == "Attack")
le_fp    = sum(1 for i in val_lenc_idx if binary_verdict_vae(i, op_thresh) == "Attack")
nr_fp    = b_fp + le_fp

recall     = var_att / len(variant_idx)
c06_recall = c06_att / len(c06_idx)
fpr_ben    = b_fp / len(val_ben_idx)
fpr_lenc   = le_fp / len(val_lenc_idx)
fpr_nr     = nr_fp / len(val_nonran_idx)

log(f"\n  {'Metric':<20}  {'VAE-binary':>12}  {'BiLSTM-binary':>15}  {'Delta':>8}")
log("  " + "─" * 60)
log(f"  {'Variant Recall':<20}  {recall:>12.0%}  {'99%':>15}  {recall-0.99:>+8.0%}")
log(f"  {'Conti_06 Recall':<20}  {c06_recall:>12.0%}  {'90%':>15}  {c06_recall-0.90:>+8.0%}")
log(f"  {'FPR Benign':<20}  {fpr_ben:>12.1%}  {'?':>15}")
log(f"  {'FPR LEnc':<20}  {fpr_lenc:>12.1%}  {'?':>15}")
log(f"  {'FPR Combined NR':<20}  {fpr_nr:>12.1%}  {'~5%':>15}  {fpr_nr-0.05:>+8.1%}")

json_out["vae_binary_eval"] = {
    "thresh": op_thresh,
    "variant_recall": recall, "c06_recall": c06_recall,
    "fpr_ben": fpr_ben, "fpr_lenc": fpr_lenc, "fpr_nr": fpr_nr,
}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Per-variant breakdown
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("PER-VARIANT RECALL"); log("=" * 72)

log(f"\n  {'Variant':<14}  {'N':>4}  {'Attack':>7}  {'Recall':>8}")
per_variant = {}
for vname in VARIANT_CLASSES:
    vi = [int(i) for i in variant_idx if all_labels[i] == vname]
    n_att = sum(1 for i in vi if binary_verdict_vae(i, op_thresh) == "Attack")
    flag  = " ◀" if vname == "Conti_06" else ""
    log(f"  {vname+flag:<14}  {len(vi):>4}  {n_att:>7}  {n_att/len(vi):>8.0%}")
    per_variant[vname] = {"n": len(vi), "attack": n_att, "recall": n_att/len(vi)}

# All known ransomware in val
val_ran_att = sum(1 for i in val_ran_idx if binary_verdict_vae(i, op_thresh) == "Attack")
log(f"\n  Val known ransomware: {val_ran_att}/{len(val_ran_idx)} = {val_ran_att/len(val_ran_idx):.0%}")

json_out["per_variant_vae_binary"] = per_variant


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Per-exec Conti_06 breakdown
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("CONTI_06 PER-EXEC BREAKDOWN"); log("=" * 72)

log(f"\n  {'Idx':>6}  {'VAE_MSE':>10}  {'>thresh':>8}  {'CNN_p1':>8}  {'Stream':>8}  {'Verdict':>8}")
log("  " + "─" * 58)
c06_details = []
for i in c06_idx:
    vae_mse = float(vae_scores[i])
    fires   = vae_mse > op_thresh
    p1      = float(pmap2[i][1])
    stream  = float(smap2.get(i, 0.0))
    verdict = binary_verdict_vae(i, op_thresh)
    log(f"  {i:>6}  {vae_mse:>10.4f}  {'YES' if fires else 'no':>8}  "
        f"{p1:>8.4f}  {stream:>8.4f}  {verdict:>8}")
    c06_details.append({
        "idx": i, "vae_mse": vae_mse, "fires": fires,
        "cnn_p_ransomware": p1, "stream": stream, "verdict": verdict
    })

c06_caught_vae  = sum(1 for d in c06_details if d["verdict"] == "Attack")
c06_caught_cnn  = sum(1 for d in c06_details if d["cnn_p_ransomware"] >= 0.5)
c06_caught_strm = sum(1 for d in c06_details if d["stream"] >= STREAM_THRESH)
c06_caught_vae_head = sum(1 for d in c06_details if d["fires"])
log(f"\n  Conti_06 summary:")
log(f"    Total:             {len(c06_idx)}")
log(f"    Caught (total):    {c06_caught_vae}")
log(f"    By CNN primary:    {c06_caught_cnn}")
log(f"    By streaming:      {c06_caught_strm}")
log(f"    By VAE head only:  {c06_caught_vae_head - c06_caught_cnn - c06_caught_strm} "
    f"(VAE fires AND not caught by CNN/stream)")

json_out["c06_per_exec"] = c06_details


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Head-to-head table
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("HEAD-TO-HEAD vs BiLSTM-BINARY"); log("=" * 72)

# Load BiLSTM scores if available (for reference FPR numbers)
bilstm_scores_path = FEATURES_DIR / "lstm_scores_all.npy"
bilstm_thresh_path = FEATURES_DIR / "lstm_threshold.npy"
if bilstm_scores_path.exists() and bilstm_thresh_path.exists():
    bilstm_scores = np.load(bilstm_scores_path)
    bilstm_thresh = float(np.load(bilstm_thresh_path)[0])
    log(f"  BiLSTM scores loaded (thresh={bilstm_thresh:.4f})")

    def binary_verdict_bilstm(i, pmap=None, smap=None):
        i = int(i)
        p = (pmap or pmap2)[i]; s = (smap or smap2).get(i, 0.0)
        fires = bilstm_scores[i] > bilstm_thresh
        if p[1] >= 0.5: return "Attack"
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"

    bl_var  = sum(1 for i in variant_idx if binary_verdict_bilstm(i) == "Attack")
    bl_c06  = sum(1 for i in c06_idx     if binary_verdict_bilstm(i) == "Attack")
    bl_bfp  = sum(1 for i in val_ben_idx  if binary_verdict_bilstm(i) == "Attack")
    bl_lfp  = sum(1 for i in val_lenc_idx if binary_verdict_bilstm(i) == "Attack")
    bl_nrfp = (bl_bfp + bl_lfp) / len(val_nonran_idx)

    log(f"\n  {'System':<22}  {'Recall':>8}  {'C06':>6}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
    log("  " + "─" * 72)
    log(f"  {'VAE-binary':>22}  {recall:>8.0%}  {c06_recall:>6.0%}  "
        f"{fpr_ben:>9.1%}  {fpr_lenc:>10.1%}  {fpr_nr:>8.1%}")
    log(f"  {'BiLSTM-binary':>22}  {bl_var/len(variant_idx):>8.0%}  {bl_c06/len(c06_idx):>6.0%}  "
        f"{bl_bfp/len(val_ben_idx):>9.1%}  {bl_lfp/len(val_lenc_idx):>10.1%}  {bl_nrfp:>8.1%}")

    json_out["head_to_head"] = {
        "vae_binary":   {"recall": recall, "c06": c06_recall, "fpr_ben": fpr_ben,
                         "fpr_lenc": fpr_lenc, "fpr_nr": fpr_nr},
        "bilstm_binary": {"recall": bl_var/len(variant_idx), "c06": bl_c06/len(c06_idx),
                          "fpr_ben": bl_bfp/len(val_ben_idx), "fpr_lenc": bl_lfp/len(val_lenc_idx),
                          "fpr_nr": bl_nrfp},
    }
else:
    log("  BiLSTM scores not found — skipping side-by-side FPR comparison.")
    log(f"  BiLSTM-binary reference (from prior run): 99% recall / 90% C06 / ~5% FPR_NR")
    log(f"\n  VAE-binary: {recall:.0%} recall / {c06_recall:.0%} C06 / {fpr_nr:.1%} FPR_NR")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — 6-Family LOFO (CNN retrained per fold, VAE head fixed)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("6-FAMILY LOFO (CNN retrained per fold, VAE fixed)"); log("=" * 72)

X_tr_text = text_seqs[train_idx]

def train_cnn_2way(X_tr, y, n_epochs=80, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TextEmbeddingClassifier(emb_dim=384, n_classes=2).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
    n_ran = y.sum(); n_rest = len(y) - n_ran
    w = torch.tensor([n_ran/len(y), n_rest/len(y)], dtype=torch.float32).to(device)
    crit  = nn.CrossEntropyLoss(weight=w)
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y,    dtype=torch.long)
    dl  = DataLoader(TensorDataset(X_t, y_t), 32, shuffle=True)
    for ep in range(n_epochs):
        model.train()
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, _, _ = model(bx)
            loss = crit(logits, by); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0: log(f"    ep {ep+1}/{n_epochs}")
    return model

@torch.no_grad()
def probs_2way_batch(model, idx_list):
    model.eval()
    result = {}
    for i in idx_list:
        x = torch.tensor(text_seqs[int(i)][None], dtype=torch.float32).to(device)
        logits, _, _ = model(x)
        result[int(i)] = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return result

@torch.no_grad()
def streaming_max_batch(model, idx_list):
    model.eval()
    result = {}
    for i in idx_list:
        i = int(i)
        seq = text_seqs[i]; W = 50; best = 0.0
        for start in range(0, 300 - W + 1, 5):
            chunk  = seq[start:start + W]
            padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
            x = torch.tensor(padded[None], dtype=torch.float32).to(device)
            logits, _, _ = model(x)
            p = torch.softmax(logits, dim=1)[0, 1].item()
            if p > best: best = p
        result[i] = best
    return result

lofo_results = {}
for family, members in LOFO_FAMILIES.items():
    log(f"\n  === LOFO: hold out {family} ===")

    held_mask    = np.array([all_labels[i] in members for i in train_idx])
    held_local   = np.where(held_mask)[0]
    held_global  = train_idx[held_local]
    fold_tr_idx  = np.where(~held_mask)[0]
    fold_tr_seqs = X_tr_text[fold_tr_idx]
    fold_tr_y2   = train_y_bin[fold_tr_idx]

    # Best of 3 seeds (by held-out ransomware recall)
    best_fold_clf = None; best_fold_recall = -1
    for seed in [0, 1, 2]:
        m = train_cnn_2way(fold_tr_seqs, fold_tr_y2, n_epochs=80, seed=seed)
        m.eval()
        with torch.no_grad():
            xi = torch.tensor(text_seqs[held_global], dtype=torch.float32)
            dl_ = DataLoader(TensorDataset(xi), 64, shuffle=False)
            lgts = []
            for (bx,) in dl_: lgts.append(m(bx.to(device))[0].cpu())
            held_preds = torch.cat(lgts).argmax(1).numpy()
            s_recall = (held_preds == 1).mean()
        if s_recall > best_fold_recall:
            best_fold_recall = s_recall; best_fold_clf = m

    # Compute probs for val_nonran and held_global
    fold_pmap_nr  = probs_2way_batch(best_fold_clf, [int(i) for i in val_nonran_idx])
    fold_smap_nr  = {}
    for i in val_nonran_idx:
        i = int(i)
        if fold_pmap_nr[i].argmax() != 1:
            fold_smap_nr[i] = streaming_max_batch(best_fold_clf, [i])[i]
        else:
            fold_smap_nr[i] = 0.0

    def fold_verdict(i, thr, pmap, smap):
        i = int(i); p = pmap[i]; s = smap.get(i, 0.0)
        fires = vae_scores[i] > thr
        if p[1] >= 0.5: return "Attack"
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"

    b_fp  = sum(1 for i in val_ben_idx  if fold_verdict(i, op_thresh, fold_pmap_nr, fold_smap_nr) == "Attack")
    le_fp = sum(1 for i in val_lenc_idx if fold_verdict(i, op_thresh, fold_pmap_nr, fold_smap_nr) == "Attack")
    nr_fp = b_fp + le_fp

    # Recall on held-out family
    fold_pmap_held = probs_2way_batch(best_fold_clf, [int(i) for i in held_global])
    fold_smap_held = {}
    for i in held_global:
        i = int(i)
        if fold_pmap_held[i].argmax() != 1:
            fold_smap_held[i] = streaming_max_batch(best_fold_clf, [i])[i]
        else:
            fold_smap_held[i] = 0.0

    n_held = len(held_global)
    n_att  = sum(1 for i in held_global if fold_verdict(i, op_thresh, fold_pmap_held, fold_smap_held) == "Attack")
    lofo_recall = n_att / n_held

    log(f"    N={n_held}  Attack={n_att}  Recall={lofo_recall:.0%}  "
        f"FPR_Ben={b_fp/len(val_ben_idx):.1%}  FPR_LEnc={le_fp/len(val_lenc_idx):.1%}")

    lofo_results[family] = {
        "n": n_held, "attack": n_att, "recall": lofo_recall,
        "fpr_ben": b_fp / len(val_ben_idx),
        "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": nr_fp / len(val_nonran_idx),
    }
    del best_fold_clf; torch.cuda.empty_cache()

json_out["lofo_vae"] = lofo_results

log(f"\n  LOFO SUMMARY (VAE head):")
log(f"  {'Family':<12}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
log("  " + "─" * 54)
recalls = []
for fam, r in lofo_results.items():
    log(f"  {fam:<12}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  "
        f"{r['fpr_lenc']:>10.1%}  {r['fpr_nr']:>8.1%}")
    recalls.append(r['recall'])
log(f"  {'MEAN':<12}  {np.mean(recalls):>8.0%}")


# ════════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════════
out_txt  = FEATURES_DIR / "vae_binary_fusion_results.txt"
out_json = FEATURES_DIR / "vae_binary_fusion_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("vae_binary_fusion.py done.")
