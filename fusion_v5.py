"""
RanSMAP Fusion v5 — Temporal LSTM VAE Integration
==================================================
Replaces the flat MLP VAE (threshold 0.6720) with the temporal LSTM VAE
trained in temporal_vae_training.py. Tests three scoring variants:

  v5a │ Temporal VAE sequence-level MSE   │ threshold from training (0.9363)
      │   mean reconstruction error over all 300 windows × 23 features
  v5b │ Temporal VAE max-window MSE       │ calibrated on val at 95th pct
      │   max per-window error (mean over 23 features, then max over 300 windows)
      │   → designed to catch transient anomalies averaged away in v5a
  v5c │ v5a OR v5b (union of both signals)│

Fusion structure (same as v4):
  Ransomware-primary → Attack
  Legit-Enc-primary  → if stream≥0.65 OR tvae fires → Attack; else Benign
  Benign-primary     → if stream≥0.65 → Attack; if tvae fires → Idle; else Benign

Calibration: max-window threshold tuned on val NON-RANSOMWARE only (blind to variants).
Evaluation: all 70 Conti variants + 100 mix — evaluated completely blind.

Reference baseline (fusion v4, flat VAE): 84% variant recall / 1.7% FPR
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

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]
BENIGN_IDX, LENC_IDX, RAN_IDX = 0, 1, 2

STREAM_THRESH = 0.65
W_STREAM      = 50
STRIDE        = 5
N_WINDOWS     = 300

# Must match temporal_vae_training.py exactly
TVAE_HIDDEN = 64
TVAE_LATENT = 32
TVAE_LAYERS = 2
TVAE_NFEAT  = 23

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Model definitions ──────────────────────────────────────────────────────────

class TextEmbeddingClassifier(nn.Module):
    def __init__(self, seq_len=300, emb_dim=384, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
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


class TemporalVAE(nn.Module):
    """
    Exact architecture from temporal_vae_training.py.
    BiLSTM encoder (23→64 hidden, 2 layers, bidirectional) → mu/logvar (32-d).
    LSTM decoder (32 → 64 hidden, 2 layers) → linear (64→23) per timestep.
    """
    def __init__(self, n_features=TVAE_NFEAT, hidden_dim=TVAE_HIDDEN,
                 latent_dim=TVAE_LATENT, num_layers=TVAE_LAYERS):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.n_features  = n_features

        self.encoder_lstm = nn.LSTM(
            n_features, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.fc_mu     = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        self.fc_decode    = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=0.2,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)   # fwd + bwd last layer
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, seq_len=300):
        h0  = self.fc_decode(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0  = torch.zeros_like(h0)
        inp = self.fc_decode(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(inp, (h0, c0))
        return self.output_layer(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(mu, x.size(1)), mu, logvar   # deterministic (eval)


class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.X = torch.tensor(seqs, dtype=torch.float32)
    def __len__(self):   return len(self.X)
    def __getitem__(self, i): return self.X[i]


def batch_tvae_scores(sequences, batch_size=64):
    """
    sequences : (N, 300, 23) float32, already scaled with tvae scaler.
    Returns:
      seq_mse    (N,) — mean reconstruction error over all 300 windows × 23 features
      maxwin_mse (N,) — max per-window error (mean over 23 features first, then max over windows)

    The max-window score captures transient anomalies that the mean averages away.
    Conti_06 spikes to moderate entropy in a few windows; sequence-level mean comes
    out near-threshold. max-window is the principled fix.
    """
    dl = DataLoader(SeqDataset(sequences), batch_size=batch_size, shuffle=False)
    seq_out, maxwin_out = [], []
    tvae.eval()
    with torch.no_grad():
        for bx in dl:
            bx    = bx.to(device)
            mu, _ = tvae.encode(bx)
            recon = tvae.decode(mu, bx.size(1))
            sq    = (recon - bx) ** 2              # (B, 300, 23)
            pw    = sq.mean(dim=2)                  # (B, 300) — per-window mean over features
            seq_out.append(pw.mean(dim=1).cpu().numpy())           # mean over windows → (B,)
            maxwin_out.append(pw.max(dim=1).values.cpu().numpy())  # max  over windows → (B,)
    return np.concatenate(seq_out), np.concatenate(maxwin_out)


# ── Load data ──────────────────────────────────────────────────────────────────

print("\nLoading data...")
text_seqs  = np.load(FEATURES_DIR / "text_sequences.npy")
all_labels = np.load(FEATURES_DIR / "text_labels.npy")

meta_df = pd.read_csv(FEATURES_DIR / "metadata.csv")
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for idx, row in meta_df.iterrows():
    feat_23[idx] = np.load(row["filepath"])
print(f"  feat_23 : {feat_23.shape}")
print(f"  text_seqs: {text_seqs.shape}")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx     = np.where(split["mix_mask"])[0]
print(f"  Val: {len(val_idx)}  |  Variants: {len(variant_idx)}  |  Mix: {len(mix_idx)}")

print("\nLoading models...")
clf = TextEmbeddingClassifier().to(device)
clf.load_state_dict(
    torch.load(FEATURES_DIR / "classifier.pt", map_location=device, weights_only=True)
)
clf.eval()

tvae = TemporalVAE().to(device)
tvae.load_state_dict(
    torch.load(FEATURES_DIR / "temporal_vae_model.pt", map_location=device, weights_only=True)
)
tvae.eval()

tvae_mean  = np.load(FEATURES_DIR / "temporal_vae_scaler_mean.npy")
tvae_scale = np.load(FEATURES_DIR / "temporal_vae_scaler_scale.npy")
THRESH_SEQ = float(np.load(FEATURES_DIR / "temporal_vae_threshold.npy")[0])
print(f"  Temporal VAE seq-level threshold (from training): {THRESH_SEQ:.4f}")


# ── Precompute all signals ─────────────────────────────────────────────────────

eval_indices = np.concatenate([val_idx, variant_idx, mix_idx])
N_eval = len(eval_indices)

# [1/3] Primary classifier probabilities
print(f"\n[1/3] Primary probs for {N_eval} executions...")
pmap = {}
for k, idx in enumerate(eval_indices):
    i = int(idx)
    x = torch.tensor(text_seqs[i], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = clf(x)
    pmap[i] = torch.softmax(logits, dim=1).cpu().numpy()[0]
    if (k + 1) % 200 == 0:
        print(f"  {k+1}/{N_eval}", flush=True)
print(f"  Done. Ransomware-primary: {sum(1 for i in pmap if pmap[i].argmax()==RAN_IDX)}")

# [2/3] Streaming scores (only for non-Ransomware primaries — same as fusion v4)
print(f"\n[2/3] Streaming scores for non-Ransomware primaries...")
smap = {}
n_streaming = 0
for k, idx in enumerate(eval_indices):
    i = int(idx)
    if pmap[i].argmax() != RAN_IDX:
        n_streaming += 1
        seq, best = text_seqs[i], 0.0
        with torch.no_grad():
            for start in range(0, N_WINDOWS - W_STREAM + 1, STRIDE):
                chunk  = seq[start:start + W_STREAM]
                padded = np.concatenate([chunk, np.zeros((N_WINDOWS - W_STREAM, 384), np.float32)])
                x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _, _ = clf(x)
                p_ran = torch.softmax(logits, dim=1)[0, RAN_IDX].item()
                if p_ran > best:
                    best = p_ran
        smap[i] = best
    else:
        smap[i] = 0.0
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{N_eval}  (streaming computed for {n_streaming} so far)", flush=True)
print(f"  Done. Streaming computed for {n_streaming} executions.")

# [3/3] Temporal VAE scores — sequence-level AND max-window for all eval data
print(f"\n[3/3] Temporal VAE scores for all {N_eval} executions...")
eval_feat = feat_23[eval_indices]                           # (N_eval, 300, 23)
flat_feat = eval_feat.reshape(-1, 23)
scaled    = ((flat_feat - tvae_mean) / tvae_scale).reshape(N_eval, 300, 23).astype(np.float32)
eval_seq_mse, eval_maxwin_mse = batch_tvae_scores(scaled)

seq_map    = {int(eval_indices[i]): float(eval_seq_mse[i])    for i in range(N_eval)}
maxwin_map = {int(eval_indices[i]): float(eval_maxwin_mse[i]) for i in range(N_eval)}
print("  Done.")


# ── Calibrate max-window threshold on val non-ransomware ──────────────────────

val_nonran_keys = [int(i) for i in val_idx if all_labels[int(i)] not in RANSOMWARE_CLASSES]
val_seq_scores    = np.array([seq_map[i]    for i in val_nonran_keys])
val_maxwin_scores = np.array([maxwin_map[i] for i in val_nonran_keys])

THRESH_MAXWIN_90 = float(np.percentile(val_maxwin_scores, 90))
THRESH_MAXWIN_95 = float(np.percentile(val_maxwin_scores, 95))   # primary choice
THRESH_MAXWIN_97 = float(np.percentile(val_maxwin_scores, 97))
THRESH_MAXWIN_99 = float(np.percentile(val_maxwin_scores, 99))

print(f"\nCalibration on {len(val_nonran_keys)} val non-ransomware executions:")
print(f"  Seq-level   — mean={val_seq_scores.mean():.4f}  "
      f"95th={np.percentile(val_seq_scores,95):.4f}  "
      f"FPR at saved threshold ({THRESH_SEQ:.4f}): {(val_seq_scores > THRESH_SEQ).mean():.1%}")
print(f"  Max-window  — mean={val_maxwin_scores.mean():.4f}  "
      f"90th={THRESH_MAXWIN_90:.4f}  95th={THRESH_MAXWIN_95:.4f}  "
      f"97th={THRESH_MAXWIN_97:.4f}  99th={THRESH_MAXWIN_99:.4f}")
print(f"\n  ★ Primary max-window threshold: {THRESH_MAXWIN_95:.4f} (95th pct, same philosophy as seq-level)")


# ── Fusion functions ───────────────────────────────────────────────────────────

def _tvae_fires(i, mode, st, mt):
    """Does the temporal VAE fire for execution i, given mode and thresholds?"""
    if mode == "seq":
        return seq_map[i] > st
    if mode == "maxwin":
        return maxwin_map[i] > mt
    # "both" = union
    return seq_map[i] > st or maxwin_map[i] > mt


def fuse_v5(p, s, i, mode="seq", st=THRESH_SEQ, mt=THRESH_MAXWIN_95):
    """
    p : classifier softmax [benign, legit_enc, ransomware]
    s : max streaming score
    i : execution index (int), for temporal VAE lookup
    mode: "seq" | "maxwin" | "both"
    st  : seq-level threshold
    mt  : max-window threshold
    """
    fires = _tvae_fires(i, mode, st, mt)

    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]:
        return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        if s >= STREAM_THRESH or fires:
            return "Attack"
        return "Benign"
    # Benign primary
    if s >= STREAM_THRESH:
        return "Attack"
    if fires:
        return "Idle"
    return "Benign"


# ── Helpers ───────────────────────────────────────────────────────────────────

val_nonran_set = set(val_nonran_keys)
all_variant_keys = [int(i) for i in variant_idx]
n_total_var = len(all_variant_keys)
n_total_nonran = len(val_nonran_keys)


def eval_fpr(mode, st=THRESH_SEQ, mt=THRESH_MAXWIN_95):
    attacks = sum(1 for i in val_nonran_keys
                  if fuse_v5(pmap[i], smap[i], i, mode, st, mt) == "Attack")
    return attacks, n_total_nonran


def eval_variant_recall(mode, st=THRESH_SEQ, mt=THRESH_MAXWIN_95):
    """Returns (per_variant_dict, total_detected, total_n)."""
    per = {}
    total_det = 0
    for vname in VARIANT_CLASSES:
        vi = [int(i) for i in variant_idx if all_labels[int(i)] == vname]
        det  = sum(1 for i in vi if fuse_v5(pmap[i], smap[i], i, mode, st, mt) == "Attack")
        idle = sum(1 for i in vi if fuse_v5(pmap[i], smap[i], i, mode, st, mt) == "Idle")
        per[vname] = {"n": len(vi), "det": det, "idle": idle}
        total_det += det
    return per, total_det, n_total_var


# ── Main report ───────────────────────────────────────────────────────────────

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(s)


log("=" * 76)
log("FUSION V5 — Temporal LSTM VAE Integration")
log(f"  Reference baseline (fusion v4, flat VAE): 84% recall / 1.7% FPR")
log(f"  Seq-level  threshold : {THRESH_SEQ:.4f}  (95th pct of training data)")
log(f"  Max-window threshold : {THRESH_MAXWIN_95:.4f}  (95th pct of val non-ransomware)")
log("=" * 76)

EVAL_MODES = [
    ("v5a — seq-level  (temporal VAE, mean MSE)", "seq",    THRESH_SEQ, THRESH_MAXWIN_95),
    ("v5b — max-window (temporal VAE, max MSE)",  "maxwin", THRESH_SEQ, THRESH_MAXWIN_95),
    ("v5c — seq OR max (union)",                  "both",   THRESH_SEQ, THRESH_MAXWIN_95),
]

json_results = {}

for label, mode, st, mt in EVAL_MODES:
    log(f"\n{'─'*76}")
    log(f"  {label}")
    log(f"{'─'*76}")

    # VAL FPR
    nr_atk, nr_n = eval_fpr(mode, st, mt)
    log(f"\n  VAL FPR (non-ransomware):")
    log(f"    Combined      {nr_atk}/{nr_n} ({100*nr_atk/nr_n:.1f}%)")
    for cls_name, cls_list in [("Benign", BENIGN_CLASSES), ("Legit-Enc", LEGIT_ENC_CLASSES)]:
        cls_keys = [int(i) for i in val_idx if all_labels[int(i)] in cls_list]
        atk = sum(1 for i in cls_keys if fuse_v5(pmap[i], smap[i], i, mode, st, mt) == "Attack")
        log(f"      {cls_name:<12} {atk}/{len(cls_keys)} ({100*atk/len(cls_keys):.1f}%)")

    # VARIANT RECALL
    per_var, total_det, n_var = eval_variant_recall(mode, st, mt)
    log(f"\n  VARIANT RECALL:")
    log(f"  {'Variant':<14}  {'N':>4}  {'Det':>5}  {'Recall':>7}  {'AvgSeqMSE':>11}  {'AvgMaxWin':>11}")
    log(f"  {'─'*60}")
    for vname in VARIANT_CLASSES:
        if vname not in per_var: continue
        d = per_var[vname]
        vi = [int(i) for i in variant_idx if all_labels[int(i)] == vname]
        avg_seq = np.mean([seq_map[i]    for i in vi]) if vi else 0.0
        avg_mw  = np.mean([maxwin_map[i] for i in vi]) if vi else 0.0
        flag = " ◀" if vname == "Conti_06" else ""
        log(f"  {(vname+flag):<14}  {d['n']:>4}  {d['det']:>3}(+{d['idle']}I)  "
            f"{100*d['det']/d['n']:>6.0f}%  {avg_seq:>11.4f}  {avg_mw:>11.4f}")
    log(f"  {'─'*60}")
    log(f"  {'ALL':<14}  {n_var:>4}  {total_det:>5}          {100*total_det/n_var:>6.0f}%")

    # MIX SET
    if len(mix_idx) > 0:
        log(f"\n  MIX SET:")
        mix_labels_unique = sorted(set(all_labels[int(i)] for i in mix_idx))
        for mc in mix_labels_unique[:8]:
            mc_keys = [int(i) for i in mix_idx if all_labels[int(i)] == mc]
            atk = sum(1 for i in mc_keys if fuse_v5(pmap[i], smap[i], i, mode, st, mt) == "Attack")
            log(f"    {mc:<35} {atk}/{len(mc_keys)} ({100*atk/len(mc_keys):.0f}%)")

    json_results[mode] = {
        "label": label,
        "recall": total_det / n_var,
        "fpr": nr_atk / nr_n,
        "per_variant": {
            vn: {"n": v["n"], "det": v["det"], "recall": v["det"]/v["n"]}
            for vn, v in per_var.items() if v["n"] > 0
        }
    }


# ── Conti_06 per-execution deep-dive ──────────────────────────────────────────

log(f"\n{'='*76}")
log("CONTI_06 PER-EXECUTION DETAIL")
log(f"Thresholds — seq: {THRESH_SEQ:.4f}  |  max-win: {THRESH_MAXWIN_95:.4f}")
log(f"{'='*76}")
c06_keys = [int(i) for i in variant_idx if all_labels[int(i)] == "Conti_06"]

log(f"{'Exec':>6}  {'Primary':>10}  {'Stream':>8}  {'SeqMSE':>8}  {'MaxWin':>8}  {'V5a':>7}  {'V5b':>7}  {'V5c':>7}")
log("─" * 76)

for i in sorted(c06_keys):
    p   = pmap[i]
    s   = smap[i]
    pri = ["Benign", "LegEnc", "Ran"][p.argmax()]
    sm  = seq_map[i]
    mw  = maxwin_map[i]
    r5a = fuse_v5(p, s, i, "seq",    THRESH_SEQ, THRESH_MAXWIN_95)
    r5b = fuse_v5(p, s, i, "maxwin", THRESH_SEQ, THRESH_MAXWIN_95)
    r5c = fuse_v5(p, s, i, "both",   THRESH_SEQ, THRESH_MAXWIN_95)
    log(f"{i:>6}  {pri:>10}  {s:>8.3f}  {sm:>8.4f}  {mw:>8.4f}  {r5a:>7}  {r5b:>7}  {r5c:>7}")

log()
c06_seq_above = sum(1 for i in c06_keys if seq_map[i] > THRESH_SEQ)
c06_mw_above  = sum(1 for i in c06_keys if maxwin_map[i] > THRESH_MAXWIN_95)
log(f"Conti_06 seq_mse > {THRESH_SEQ:.4f}  : {c06_seq_above}/10")
log(f"Conti_06 maxwin  > {THRESH_MAXWIN_95:.4f}  : {c06_mw_above}/10")
log(f"Conti_06 seq_mse stats — min={min(seq_map[i] for i in c06_keys):.4f}  "
    f"mean={np.mean([seq_map[i] for i in c06_keys]):.4f}  "
    f"max={max(seq_map[i] for i in c06_keys):.4f}")
log(f"Conti_06 maxwin  stats — min={min(maxwin_map[i] for i in c06_keys):.4f}  "
    f"mean={np.mean([maxwin_map[i] for i in c06_keys]):.4f}  "
    f"max={max(maxwin_map[i] for i in c06_keys):.4f}")


# ── Max-window threshold sensitivity sweep ────────────────────────────────────

log(f"\n{'='*76}")
log("MAX-WINDOW THRESHOLD SENSITIVITY (v5b — variant recall + FPR + Conti_06)")
log(f"Val non-ransomware n={n_total_nonran} | Total variants n={n_total_var}")
log(f"{'='*76}")
log(f"  {'Threshold':>10}  {'Pct':>6}  {'Recall':>8}  {'FPR':>8}  {'C06 det':>10}")
log(f"  {'─'*52}")

for pct in [85, 88, 90, 92, 95, 97, 99]:
    t = float(np.percentile(val_maxwin_scores, pct))
    det = sum(1 for i in all_variant_keys
              if fuse_v5(pmap[i], smap[i], i, "maxwin", THRESH_SEQ, t) == "Attack")
    fpr_n = sum(1 for i in val_nonran_keys
                if fuse_v5(pmap[i], smap[i], i, "maxwin", THRESH_SEQ, t) == "Attack")
    c06d  = sum(1 for i in c06_keys
                if fuse_v5(pmap[i], smap[i], i, "maxwin", THRESH_SEQ, t) == "Attack")
    marker = "  ← primary" if pct == 95 else ""
    log(f"  {t:>10.4f}  {pct:>5}th  {100*det/n_total_var:>7.0f}%  {100*fpr_n/n_total_nonran:>7.1f}%  "
        f"{c06d:>3}/10 ({100*c06d/10:.0f}%){marker}")


# ── Score distribution summary (for Opus) ────────────────────────────────────

log(f"\n{'='*76}")
log("SCORE DISTRIBUTIONS BY GROUP (for threshold interpretation)")
log(f"{'='*76}")

groups_to_check = [
    ("Val Benign",   [int(i) for i in val_idx if all_labels[int(i)] in BENIGN_CLASSES]),
    ("Val Legit-Enc",[int(i) for i in val_idx if all_labels[int(i)] in LEGIT_ENC_CLASSES]),
    ("Val Ransomware",[int(i) for i in val_idx if all_labels[int(i)] in RANSOMWARE_CLASSES]),
]
for vname in VARIANT_CLASSES:
    vi = [int(i) for i in variant_idx if all_labels[int(i)] == vname]
    if vi:
        groups_to_check.append((vname, vi))

log(f"  {'Group':<20}  {'N':>5}  {'SeqMSE mean':>12}  {'SeqMSE 95th':>12}  {'MaxWin mean':>12}  {'MaxWin 95th':>12}")
log(f"  {'─'*78}")
for gname, gkeys in groups_to_check:
    if not gkeys: continue
    sm_arr = np.array([seq_map[i]    for i in gkeys])
    mw_arr = np.array([maxwin_map[i] for i in gkeys])
    flag = " ◀" if gname == "Conti_06" else ""
    log(f"  {(gname+flag):<20}  {len(gkeys):>5}  {sm_arr.mean():>12.4f}  "
        f"{np.percentile(sm_arr,95):>12.4f}  {mw_arr.mean():>12.4f}  {np.percentile(mw_arr,95):>12.4f}")


# ── Verdict ───────────────────────────────────────────────────────────────────

log(f"\n{'='*76}")
log("VERDICT")
log(f"{'='*76}")
log(f"  Reference (fusion v4, flat MLP VAE):         84% recall  /  1.7% FPR")

for label, mode, st, mt in EVAL_MODES:
    det = sum(1 for i in all_variant_keys if fuse_v5(pmap[i], smap[i], i, mode, st, mt) == "Attack")
    fpr_n = sum(1 for i in val_nonran_keys if fuse_v5(pmap[i], smap[i], i, mode, st, mt) == "Attack")
    log(f"  {label:<46}  {100*det/n_total_var:.0f}% recall  /  {100*fpr_n/n_total_nonran:.1f}% FPR")

log()


# ── Save outputs ──────────────────────────────────────────────────────────────

summary = "\n".join(lines)
(FEATURES_DIR / "fusion_v5_results.txt").write_text(summary)

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        return super().default(o)

full_json = {
    "version": "v5",
    "reference_v4": {"recall": 0.84, "fpr": 0.017},
    "thresholds": {
        "seq_level":    THRESH_SEQ,
        "maxwin_90":    THRESH_MAXWIN_90,
        "maxwin_95":    THRESH_MAXWIN_95,
        "maxwin_97":    THRESH_MAXWIN_97,
        "maxwin_99":    THRESH_MAXWIN_99,
    },
    "modes": json_results,
    "conti_06_scores": {
        str(i): {"seq_mse": seq_map[i], "maxwin_mse": maxwin_map[i]}
        for i in sorted(c06_keys)
    },
}
with open(FEATURES_DIR / "fusion_v5_results.json", "w") as f:
    json.dump(full_json, f, indent=2, cls=NpEncoder)

print(f"\nSaved: fusion_v5_results.txt / .json")
print("Done.")
