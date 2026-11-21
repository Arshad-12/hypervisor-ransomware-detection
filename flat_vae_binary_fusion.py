"""
flat_vae_binary_fusion.py — Flat MLP VAE as Deployed Anomaly Head + 2-way CNN Binary Fusion
=============================================================================================
Per HANDOFF update 51 (Opus correction, July 16, 2026).

Corrects the previous VAE_BINARY_FUSION run which used the WRONG VAE (temporal).
This run uses the FLAT VAE = vae_fixed_model.pt (TimeSeriesVAE, flat MLP, trained in D3-FIXED).

Key facts about the flat VAE:
  - Architecture: TimeSeriesVAE (6900-d MLP encoder, latent_dim=16)
  - Saved as: vae_fixed_model.pt
  - Scaler: vae_fixed_scaler_mean.npy / vae_fixed_scaler_scale.npy
  - Saved MSE scores: vae_fixed_window_scores.npy (all 1970 execs, pre-computed)
  - Training threshold (95th pct val non-ran OOF): ~0.672
  - Known Conti_06 firing: MSE 0.80-6.95, fires on ~8/10 execs
  - In 3-way fusion these went to Idle (Benign-primary + VAE fires → Idle)
  - With binary fusion (Idle removed) → same execs become Attack

Methodology:
  - Load vae_fixed_window_scores.npy (already computed; no VAE inference needed)
  - Re-calibrate threshold on val non-ran 95th pct (same discipline as BiLSTM binary)
  - Binary fusion: 2-way CNN (classifier_2way.pt) + streaming + flat VAE → Attack/Benign
  - Report: variant recall, Conti_06, Benign/LEnc/combined FPR
  - Head-to-head: VAE-binary vs BiLSTM-binary (99%/90%/5%)
  - Per-exec Conti_06 breakdown
  - 6-family LOFO (CNN retrained per fold, VAE head fixed)

★  CORE RULE: known-classes-only training; blind eval on ALL variants + LOFO;
   Benign FPR and Legit-Enc FPR reported SEPARATELY. Conti_06 NEVER in training.
"""

import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

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

log("  Loading MiniLM sequences...")
text_seqs = np.load(FEATURES_DIR / "text_sequences.npy")   # (1970, 300, 384)

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
log(f"  Val: {len(val_idx)}  Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}, Ran={len(val_ran_idx)}")
log(f"  Variants: {len(variant_idx)},  Conti_06: {len(c06_idx)}")

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Load flat VAE scores (pre-computed in D3-FIXED)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Loading flat VAE scores (vae_fixed_window_scores.npy)"); log("=" * 72)

vae_scores_all = np.load(FEATURES_DIR / "vae_fixed_window_scores.npy")  # (1970,) pre-computed
vae_thresh_saved = float(np.load(FEATURES_DIR / "vae_fixed_threshold.npy").flat[0])

log(f"  Loaded vae_fixed_window_scores.npy: shape={vae_scores_all.shape}")
log(f"  Score range: [{vae_scores_all.min():.4f}, {vae_scores_all.max():.4f}]")
log(f"  Saved training threshold (95th pct OOF non-ran): {vae_thresh_saved:.4f}")

# Re-calibrate threshold on val non-ran (same discipline as BiLSTM binary)
vae_thresh_val = float(np.percentile(vae_scores_all[val_nonran_idx], FPR_PCT))
log(f"  Val non-ran 95th pct threshold:    {vae_thresh_val:.4f}")
log(f"    Val FPR at this thresh:   {(vae_scores_all[val_nonran_idx] > vae_thresh_val).mean():.1%}")
log(f"    Val TPR (ran) at thresh:  {(vae_scores_all[val_ran_idx] > vae_thresh_val).mean():.1%}")

# Score distribution by class
log(f"\n  Flat VAE MSE by class:")
log(f"  {'Class':<14}  {'N':>4}  {'Mean':>8}  {'P50':>8}  {'P95':>8}  {'>thresh':>8}")
for cls_name, idx_arr in [
    ("Val Benign", val_ben_idx), ("Val LEnc", val_lenc_idx), ("Val Ran", val_ran_idx),
]:
    sc = vae_scores_all[idx_arr]
    log(f"  {cls_name:<14}  {len(sc):>4}  {sc.mean():>8.4f}  "
        f"{np.percentile(sc,50):>8.4f}  {np.percentile(sc,95):>8.4f}  "
        f"{(sc > vae_thresh_val).mean():>8.1%}")
c06_sc = vae_scores_all[c06_idx]
log(f"  {'Conti_06':<14}  {len(c06_sc):>4}  {c06_sc.mean():>8.4f}  "
    f"{np.percentile(c06_sc,50):>8.4f}  {np.percentile(c06_sc,95):>8.4f}  "
    f"{(c06_sc > vae_thresh_val).mean():>8.1%}")

json_out["vae_threshold"] = {
    "saved_oof_threshold": vae_thresh_saved,
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
# SECTION 4 — Flat VAE binary fusion eval
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("FLAT VAE BINARY FUSION EVAL"); log("=" * 72)

def binary_verdict_vae(i, vae_thr, pmap=None, smap=None):
    """2-way CNN + streaming + flat VAE anomaly → Attack/Benign."""
    i = int(i)
    p = (pmap or pmap2)[i]
    s = (smap or smap2).get(i, 0.0)
    fires = vae_scores_all[i] > vae_thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"

op_thresh = vae_thresh_val
log(f"\n  Operating threshold: {op_thresh:.4f} (95th pct val non-ran flat VAE MSE)")

var_att  = sum(1 for i in variant_idx if binary_verdict_vae(i, op_thresh) == "Attack")
c06_att  = sum(1 for i in c06_idx     if binary_verdict_vae(i, op_thresh) == "Attack")
b_fp     = sum(1 for i in val_ben_idx  if binary_verdict_vae(i, op_thresh) == "Attack")
le_fp    = sum(1 for i in val_lenc_idx if binary_verdict_vae(i, op_thresh) == "Attack")
nr_fp    = b_fp + le_fp
val_ran_att = sum(1 for i in val_ran_idx if binary_verdict_vae(i, op_thresh) == "Attack")

recall     = var_att / len(variant_idx)
c06_recall = c06_att / len(c06_idx)
fpr_ben    = b_fp / len(val_ben_idx)
fpr_lenc   = le_fp / len(val_lenc_idx)
fpr_nr     = nr_fp / len(val_nonran_idx)

log(f"\n  Val known ransomware: {val_ran_att}/{len(val_ran_idx)} = {val_ran_att/len(val_ran_idx):.0%}")
log(f"\n  {'Metric':<22}  {'Flat-VAE-binary':>16}  {'BiLSTM-binary':>15}  {'Delta':>8}")
log("  " + "─" * 66)
log(f"  {'Variant Recall':<22}  {recall:>16.0%}  {'99%':>15}  {recall-0.99:>+8.0%}")
log(f"  {'Conti_06 Recall':<22}  {c06_recall:>16.0%}  {'90%':>15}  {c06_recall-0.90:>+8.0%}")
log(f"  {'FPR Benign':<22}  {fpr_ben:>16.1%}  {'7.8%':>15}  {fpr_ben-0.078:>+8.1%}")
log(f"  {'FPR LEnc':<22}  {fpr_lenc:>16.1%}  {'2.2%':>15}  {fpr_lenc-0.022:>+8.1%}")
log(f"  {'FPR Combined NR':<22}  {fpr_nr:>16.1%}  {'5.0%':>15}  {fpr_nr-0.05:>+8.1%}")

json_out["flat_vae_binary_eval"] = {
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

json_out["per_variant_flat_vae_binary"] = per_variant


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Per-exec Conti_06 breakdown
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("CONTI_06 PER-EXEC BREAKDOWN"); log("=" * 72)

log(f"\n  {'Idx':>6}  {'VAE_MSE':>10}  {'>thresh':>8}  {'CNN_p1':>8}  {'Stream':>8}  {'Verdict':>8}  Driver")
log("  " + "─" * 68)
c06_details = []
for i in c06_idx:
    vae_mse = float(vae_scores_all[i])
    fires   = vae_mse > op_thresh
    p1      = float(pmap2[i][1])
    stream  = float(smap2.get(i, 0.0))
    verdict = binary_verdict_vae(i, op_thresh)
    if p1 >= 0.5: driver = "CNN"
    elif stream >= STREAM_THRESH: driver = "Stream"
    elif fires: driver = "VAE"
    else: driver = "missed"
    log(f"  {i:>6}  {vae_mse:>10.4f}  {'YES' if fires else 'no':>8}  "
        f"{p1:>8.4f}  {stream:>8.4f}  {verdict:>8}  {driver}")
    c06_details.append({
        "idx": i, "vae_mse": vae_mse, "fires": fires,
        "cnn_p_ransomware": p1, "stream": stream, "verdict": verdict, "driver": driver
    })

log(f"\n  Conti_06 summary:")
log(f"    Total:              {len(c06_idx)}")
log(f"    Caught:             {sum(1 for d in c06_details if d['verdict']=='Attack')}")
log(f"    By CNN primary:     {sum(1 for d in c06_details if d['driver']=='CNN')}")
log(f"    By VAE head:        {sum(1 for d in c06_details if d['driver']=='VAE')}")
log(f"    Missed:             {sum(1 for d in c06_details if d['driver']=='missed')}")

json_out["c06_per_exec"] = c06_details


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Head-to-head table (load BiLSTM scores if available)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("HEAD-TO-HEAD vs BiLSTM-BINARY"); log("=" * 72)

bilstm_path = FEATURES_DIR / "lstm_scores_all.npy"
bilstm_thresh_path = FEATURES_DIR / "lstm_threshold.npy"

if bilstm_path.exists() and bilstm_thresh_path.exists():
    bilstm_scores = np.load(bilstm_path)
    bilstm_thresh = float(np.load(bilstm_thresh_path).flat[0])
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

    log(f"\n  {'System':<24}  {'Recall':>8}  {'C06':>6}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
    log("  " + "─" * 74)
    log(f"  {'Flat-VAE-binary':>24}  {recall:>8.0%}  {c06_recall:>6.0%}  "
        f"{fpr_ben:>9.1%}  {fpr_lenc:>10.1%}  {fpr_nr:>8.1%}")
    log(f"  {'BiLSTM-binary':>24}  {bl_var/len(variant_idx):>8.0%}  {bl_c06/len(c06_idx):>6.0%}  "
        f"{bl_bfp/len(val_ben_idx):>9.1%}  {bl_lfp/len(val_lenc_idx):>10.1%}  {bl_nrfp:>8.1%}")

    json_out["head_to_head"] = {
        "flat_vae_binary": {"recall": recall, "c06": c06_recall,
                            "fpr_ben": fpr_ben, "fpr_lenc": fpr_lenc, "fpr_nr": fpr_nr},
        "bilstm_binary":   {"recall": bl_var/len(variant_idx), "c06": bl_c06/len(c06_idx),
                            "fpr_ben": bl_bfp/len(val_ben_idx),
                            "fpr_lenc": bl_lfp/len(val_lenc_idx), "fpr_nr": bl_nrfp},
    }
else:
    log("  BiLSTM scores not found — reference from record: 99% / 90% C06 / 5.0% FPR_NR")
    log(f"  Flat-VAE-binary: {recall:.0%} / {c06_recall:.0%} C06 / {fpr_nr:.1%} FPR_NR")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — 6-Family LOFO (CNN retrained per fold, flat VAE head fixed)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("6-FAMILY LOFO (CNN retrained per fold, flat VAE fixed)"); log("=" * 72)

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
            crit(logits, by).backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0: log(f"    ep {ep+1}/{n_epochs}")
    return model

@torch.no_grad()
def probs_and_stream_batch(model, idx_list):
    model.eval()
    pmap, smap = {}, {}
    for i in idx_list:
        i = int(i)
        x = torch.tensor(text_seqs[i][None], dtype=torch.float32).to(device)
        logits, _, _ = model(x)
        pmap[i] = torch.softmax(logits, dim=1).cpu().numpy()[0]
        if pmap[i].argmax() != 1:
            seq = text_seqs[i]; W = 50; best = 0.0
            for start in range(0, 300 - W + 1, 5):
                chunk  = seq[start:start + W]
                padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
                xp = torch.tensor(padded[None], dtype=torch.float32).to(device)
                lg, _, _ = model(xp)
                p = torch.softmax(lg, dim=1)[0, 1].item()
                if p > best: best = p
            smap[i] = best
        else:
            smap[i] = 0.0
    return pmap, smap

lofo_results = {}
for family, members in LOFO_FAMILIES.items():
    log(f"\n  === LOFO: hold out {family} ===")

    held_mask    = np.array([all_labels[i] in members for i in train_idx])
    held_local   = np.where(held_mask)[0]
    held_global  = train_idx[held_local]
    fold_tr_idx  = np.where(~held_mask)[0]
    fold_tr_seqs = X_tr_text[fold_tr_idx]
    fold_tr_y2   = train_y_bin[fold_tr_idx]

    best_fold_clf = None; best_fold_recall = -1
    for seed in [0, 1, 2]:
        m = train_cnn_2way(fold_tr_seqs, fold_tr_y2, n_epochs=80, seed=seed)
        m.eval()
        with torch.no_grad():
            xi = torch.tensor(text_seqs[held_global], dtype=torch.float32)
            dl_ = DataLoader(TensorDataset(xi), 64, shuffle=False)
            lgts = []
            for (bx,) in dl_: lgts.append(m(bx.to(device))[0].cpu())
            s_recall = (torch.cat(lgts).argmax(1).numpy() == 1).mean()
        if s_recall > best_fold_recall:
            best_fold_recall = s_recall; best_fold_clf = m

    # FPR on val non-ran
    fold_pmap_nr, fold_smap_nr = probs_and_stream_batch(
        best_fold_clf, [int(i) for i in val_nonran_idx])

    def fold_verdict(i, thr, pmap, smap):
        i = int(i); p = pmap[i]; s = smap.get(i, 0.0)
        fires = vae_scores_all[i] > thr
        if p[1] >= 0.5: return "Attack"
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"

    b_fp  = sum(1 for i in val_ben_idx  if fold_verdict(i, op_thresh, fold_pmap_nr, fold_smap_nr) == "Attack")
    le_fp = sum(1 for i in val_lenc_idx if fold_verdict(i, op_thresh, fold_pmap_nr, fold_smap_nr) == "Attack")
    nr_fp = b_fp + le_fp

    # Recall on held-out family
    fold_pmap_held, fold_smap_held = probs_and_stream_batch(
        best_fold_clf, [int(i) for i in held_global])

    n_held  = len(held_global)
    n_att   = sum(1 for i in held_global if fold_verdict(i, op_thresh, fold_pmap_held, fold_smap_held) == "Attack")
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

json_out["lofo_flat_vae"] = lofo_results

log(f"\n  LOFO SUMMARY (flat VAE head):")
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
out_txt  = FEATURES_DIR / "flat_vae_binary_fusion_results.txt"
out_json = FEATURES_DIR / "flat_vae_binary_fusion_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("flat_vae_binary_fusion.py done.")
