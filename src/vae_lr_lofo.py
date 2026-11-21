"""
vae_lr_lofo.py — 6-Family LOFO for the VAE Likelihood-Ratio Score
====================================================================================
Per HANDOFF item 56, TASK A (highest priority; August 10, 2026).

Item 55 Task C found the likelihood-ratio score — MSE(non-ran-only VAE) -
MSE(ransomware-only VAE) — gives 94% variant recall / 70% Conti_06, the best
generative result in the project. But it has no cross-family (LOFO) number
yet, and every single-VAE formulation tried so far (all-classes, non-ran-
only, ransomware-only) has landed at LOFO mean 70%. This script closes that
gap.

Per fold: hold the family out entirely, retrain BOTH VAEs (+ their scalers)
and the CNN:
  - non-ran-only VAE: its training pool (900 non-ransomware executions)
    never contains any ransomware family, so this retrain reproduces the
    same distribution every fold — retrained anyway per spec, for
    procedural symmetry with the ransomware-only side and to avoid relying
    on a stale saved artifact.
  - ransomware-only VAE: training pool IS ransomware, so the held-out
    family's executions are excluded (750 of 900 remain).
  - CNN: standard 3-seed pick-best retrain, held-out family excluded.
Recompute the LR score for all 1970 execs, threshold = 95th pct of
validation non-ransomware LR score (standard direction — same as item 55
Task C), evaluate the held-out family's recall + paired FPRs.

★  LABEL-INFORMED — this uses a ransomware-only model, so it is NOT an
   unsupervised result (see HANDOFF item 56's labelling rule). Report
   ordering in the write-up: unsupervised VAE (89/30) -> unsupervised
   TS-TCC (93/50) -> supervised/label-informed LR (94/70/?).
★  CORE RULE: held-out family never in either VAE's training pool or the
   CNN's training pool for that fold; threshold from validation non-
   ransomware only; blind eval; FPR always reported.
"""

import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES_DIR  = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT       = 95.0
STREAM_THRESH = 0.65
N_EPOCHS      = 80

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
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

log("  Loading feat_23 (1970, 300, 23)...")
t0 = time.time()
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])
log(f"  Loaded in {time.time()-t0:.1f}s")

log("  Loading MiniLM sequences for the per-fold 2-way CNN...")
text_seqs = np.load(FEATURES_DIR / "text_sequences.npy")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]
train_y_bin = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in train_idx])

val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])

log(f"  Training pool: {len(train_idx)} (ran={train_y_bin.sum()}, non-ran={(train_y_bin==0).sum()})")
log(f"  Val: {len(val_idx)}  Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}")

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 2-way CNN class (retrained per fold) + probs/streaming helper
# ════════════════════════════════════════════════════════════════════════════════

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

@torch.no_grad()
def probs_and_stream_batch(model, idx_list):
    model.eval()
    pmap, smap = {}, {}
    for i in idx_list:
        i = int(i)
        pmap[i] = probs_2way(model, text_seqs, i)
        smap[i] = 0.0 if pmap[i].argmax() == 1 else streaming_max_2way(model, text_seqs, i)
    return pmap, smap

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


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VAE (identical architecture throughout the VAE_* line of scripts)
# ════════════════════════════════════════════════════════════════════════════════

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
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

def vae_loss_fn(recon, x, mu, logvar, beta=1.0):
    mse = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return mse + beta * kld

def train_vae(X_seqs_scaled, n_epochs=N_EPOCHS, lr=1e-3, batch_size=64, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TimeSeriesVAE().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    X_t   = torch.tensor(X_seqs_scaled, dtype=torch.float32)
    dl    = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(n_epochs):
        for (bx,) in dl:
            bx = bx.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(bx)
            vae_loss_fn(recon, bx, mu, logvar).backward()
            opt.step()
        sched.step()
    return model

@torch.no_grad()
def vae_mse_batch(model, X_seqs_scaled):
    model.eval()
    X_t = torch.tensor(X_seqs_scaled, dtype=torch.float32)
    dl  = DataLoader(TensorDataset(X_t), batch_size=64, shuffle=False)
    mse_list = []
    for (bx,) in dl:
        bx = bx.to(device)
        mu, _ = model.encode(bx)
        recon  = model.decode(mu)
        mse_list.append(F.mse_loss(recon, bx, reduction="none").mean(dim=[1, 2]).cpu().numpy())
    return np.concatenate(mse_list)

def fit_scaler_and_scale_all(raw_subset, full_feat_23):
    scaler = StandardScaler()
    scaler.fit(raw_subset.reshape(-1, 23))
    def scale(seqs):
        N, T, C = seqs.shape
        return (scaler.transform(seqs.reshape(-1, C)).astype(np.float32).reshape(N, T, C))
    return scaler, scale(raw_subset), scale(full_feat_23)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Fusion verdict (standard direction — LR score, high = Attack)
# ════════════════════════════════════════════════════════════════════════════════

def binary_verdict(i, thr, scores, pmap, smap):
    i = int(i)
    p = pmap[i]; s = smap.get(i, 0.0)
    fires = scores[i] > thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — 6-Family LOFO
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("6-FAMILY LOFO — VAE likelihood ratio (both VAEs + CNN retrained per fold)"); log("=" * 72)

X_tr_text = text_seqs[train_idx]
lofo_results = {}

for family, members in LOFO_FAMILIES.items():
    t0 = time.time()
    log(f"\n  === LOFO: hold out {family} ===")
    held_mask   = np.array([all_labels[i] in members for i in train_idx])
    held_global = train_idx[np.where(held_mask)[0]]
    fold_tr_idx = train_idx[np.where(~held_mask)[0]]
    fold_tr_y2  = train_y_bin[np.where(~held_mask)[0]]

    # -- Fold CNN (3-seed pick-best) --
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

    # -- Fold non-ran-only VAE (its pool never contains ransomware; retrained per spec anyway) --
    fold_nonran_mask   = np.array([all_labels[i] not in RANSOMWARE_CLASSES for i in fold_tr_idx])
    fold_nonran_global = fold_tr_idx[fold_nonran_mask]
    _, X_nonran_fold_scaled, X_all_nonran_scaled = fit_scaler_and_scale_all(feat_23[fold_nonran_global], feat_23)
    nonran_model = train_vae(X_nonran_fold_scaled, seed=0)
    nonran_mse_all = vae_mse_batch(nonran_model, X_all_nonran_scaled)

    # -- Fold ransomware-only VAE (held-out family excluded) --
    fold_ran_mask   = np.array([all_labels[i] in RANSOMWARE_CLASSES for i in fold_tr_idx])
    fold_ran_global = fold_tr_idx[fold_ran_mask]
    _, X_ran_fold_scaled, X_all_ran_scaled = fit_scaler_and_scale_all(feat_23[fold_ran_global], feat_23)
    ranonly_model = train_vae(X_ran_fold_scaled, seed=0)
    ranonly_mse_all = vae_mse_batch(ranonly_model, X_all_ran_scaled)

    # -- Likelihood ratio --
    lr_scores = nonran_mse_all - ranonly_mse_all
    lr_thr = float(np.percentile(lr_scores[val_nonran_idx], FPR_PCT))

    pmap_held, smap_held = probs_and_stream_batch(best_clf, [int(i) for i in held_global])
    pmap_nr,   smap_nr   = probs_and_stream_batch(best_clf, [int(i) for i in val_nonran_idx])

    n_att = sum(1 for i in held_global if binary_verdict(i, lr_thr, lr_scores, pmap_held, smap_held) == "Attack")
    b_fp  = sum(1 for i in val_ben_idx  if binary_verdict(i, lr_thr, lr_scores, pmap_nr, smap_nr) == "Attack")
    le_fp = sum(1 for i in val_lenc_idx if binary_verdict(i, lr_thr, lr_scores, pmap_nr, smap_nr) == "Attack")

    n_held = len(held_global)
    recall = n_att / n_held
    log(f"    LR thr={lr_thr:.4f}  N={n_held}  Attack={n_att}  Recall={recall:.0%}  "
        f"FPR_Ben={b_fp/len(val_ben_idx):.1%}  FPR_LEnc={le_fp/len(val_lenc_idx):.1%}  ({time.time()-t0:.0f}s)")

    lofo_results[family] = {
        "n": n_held, "attack": n_att, "recall": recall, "lr_thr": lr_thr,
        "fpr_ben": b_fp / len(val_ben_idx), "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": (b_fp + le_fp) / len(val_nonran_idx),
    }
    del best_clf, nonran_model, ranonly_model; torch.cuda.empty_cache()

json_out["lofo_lr"] = lofo_results

log(f"\n  LOFO SUMMARY — VAE likelihood ratio (both VAEs + CNN retrained per fold)")
log(f"  {'Family':<12}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
log("  " + "─" * 54)
recalls = []
for fam, r in lofo_results.items():
    log(f"  {fam:<12}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  "
        f"{r['fpr_lenc']:>10.1%}  {r['fpr_nr']:>8.1%}")
    recalls.append(r["recall"])
log(f"  {'MEAN':<12}  {np.mean(recalls):>8.0%}")
log(f"\n  Reference — BiLSTM LOFO: 100%. TS-TCC-unsupervised LOFO: 85-86%.")
log(f"              All three single-VAE formulations LOFO: 70% (all-classes, non-ran-only, ransomware-only).")
log(f"              Main-run (within-family) LR: 94% recall / 70% Conti_06 (item 55 Task C).")

out_txt  = FEATURES_DIR / "vae_lr_lofo_results.txt"
out_json = FEATURES_DIR / "vae_lr_lofo_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("vae_lr_lofo.py done.")
