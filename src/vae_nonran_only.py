"""
vae_nonran_only.py — VAE Retrained on NON-RANSOMWARE ONLY (Training-Pool Correction)
============================================================================================
Per HANDOFF item 54 (Opus + Arshad finding, August 10, 2026).

The deployed flat VAE (`vae_fixed_model.pt`, from `d3_fixed.py`) was trained
on the ENTIRE 1,800-execution training pool — all 12 known classes,
RANSOMWARE INCLUDED. `train_nonran_mask` exists in `d3_fixed.py` and IS used
for the Mahalanobis/KNN heads, but was never applied to the VAE — the VAE's
`X_tr_raw = feat_23[train_idx]` and its scaler are both fit on the full pool.
That makes it a "novelty vs. all known behaviour" detector, not the one-class
"model of normal" the report describes, and it teaches the VAE to reconstruct
ransomware well — the opposite of what an anomaly-scoring generative model
should do.

This script changes exactly ONE thing vs. `d3_fixed.py`'s VAE section: the
VAE (and its StandardScaler) train on `train_idx[train_nonran_mask]` —
non-ransomware known classes only (Firefox, Office, Idle, AESCrypt, SDelete,
Zip; 900 executions) — instead of the full 1,800-execution pool. Everything
else (architecture, epochs, fusion, threshold discipline) is identical to
`flat_vae_binary_fusion.py` for a fair head-to-head.

Unlike the original flat-VAE run (which trained 3 seeds and kept only the
best-by-val model, reporting a single number), this script treats each of
the 3 seeds as an INDEPENDENT full pipeline (train -> threshold -> eval) and
reports mean ± std, per item 54's explicit ask and for parity with how the
TS-TCC comparisons were reported. The best-by-val seed is additionally saved
as the canonical deployable artifact (`vae_nonran_model.pt` etc.) — the
existing `vae_fixed_*` files are NOT touched, so the all-classes VAE stays
available for comparison.

LOFO note: because this VAE trains on non-ransomware classes only, holding
out a ransomware FAMILY never changes its training data or its threshold
(val_nonran_idx is also all non-ransomware) — so the VAE model is IDENTICAL
across all 6 LOFO folds. Only the 2-way CNN needs retraining per fold
(3-seed pick-best, same procedure used throughout this project).

★  CORE RULE: VAE trains on non-ransomware known classes only; Conti_06 and
   ALL ransomware (known or held-out) NEVER in VAE training; threshold from
   validation non-ransomware only; blind eval on ALL variants + LOFO; FPR
   always reported. No tuning to Conti_06.
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
SEEDS         = [0, 1, 2]
N_EPOCHS      = 80

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

log("  Loading feat_23 (1970, 300, 23)...")
t0 = time.time()
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])
log(f"  Loaded in {time.time()-t0:.1f}s")

log("  Loading MiniLM sequences for the frozen/per-fold 2-way CNN...")
text_seqs = np.load(FEATURES_DIR / "text_sequences.npy")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx     = np.where(split["mix_mask"])[0]
eval_idx    = np.concatenate([val_idx, variant_idx, mix_idx])

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]
train_y_bin = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in train_idx])
train_nonran_mask = (train_y_bin == 0)
train_nonran_global = train_idx[train_nonran_mask]   # the ONLY data the VAE sees

val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ran_idx    = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])
val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
c06_idx        = [int(i) for i in variant_idx if all_labels[i] == "Conti_06"]

log(f"  Training pool: {len(train_idx)} (ran={train_y_bin.sum()}, non-ran={train_nonran_mask.sum()})")
log(f"  VAE training set (non-ransomware ONLY): {len(train_nonran_global)} executions")
log(f"  Val: {len(val_idx)}  Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}, Ran={len(val_ran_idx)}")
log(f"  Variants: {len(variant_idx)},  Conti_06: {len(c06_idx)}")

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Frozen 2-way CNN (main run) — probs + streaming
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Loading frozen classifier_2way.pt + computing probs/streaming"); log("=" * 72)

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

log(f"  Precomputing 2-way probs + streaming for {len(eval_idx)} eval execs...")
pmap2, smap2 = {}, {}
for ii, i in enumerate(eval_idx):
    i = int(i)
    pmap2[i] = probs_2way(clf2, text_seqs, i)
    smap2[i] = 0.0 if pmap2[i].argmax() == 1 else streaming_max_2way(clf2, text_seqs, i)
    if (ii + 1) % 100 == 0: log(f"    {ii+1}/{len(eval_idx)}")
log("  Done.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VAE (identical architecture to d3_fixed.py)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("VAE definition (TimeSeriesVAE, flat MLP, latent_dim=16)"); log("=" * 72)

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


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Fit scaler on NON-RANSOMWARE TRAINING SUBSET ONLY
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Fitting StandardScaler on non-ransomware training subset ONLY"); log("=" * 72)

vae_scaler = StandardScaler()
X_nonran_raw = feat_23[train_nonran_global]          # (900, 300, 23)
vae_scaler.fit(X_nonran_raw.reshape(-1, 23))
log(f"  Scaler fit on {len(train_nonran_global)} non-ransomware executions "
    f"(Firefox/Office/Idle/AESCrypt/SDelete/Zip) — NOT the full 1800-exec pool.")

def scale_seqs(seqs):
    N, T, C = seqs.shape
    return (vae_scaler.transform(seqs.reshape(-1, C)).astype(np.float32)
            .reshape(N, T, C))

X_nonran_scaled = scale_seqs(X_nonran_raw)
X_all_scaled    = scale_seqs(feat_23)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Fusion verdict (identical to flat_vae_binary_fusion.py)
# ════════════════════════════════════════════════════════════════════════════════

def binary_verdict(i, thr, scores, pmap, smap):
    i = int(i)
    p = pmap[i]; s = smap.get(i, 0.0)
    fires = scores[i] > thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN RUN: 3 independent seeds
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("MAIN RUN — VAE trained on non-ransomware ONLY, 3 independent seeds"); log("=" * 72)

seed_metrics = []
seed_models  = {}   # seed -> (model, all_mse, thr) for saving the best-by-val one later

for seed in SEEDS:
    t0 = time.time()
    model = train_vae(X_nonran_scaled, n_epochs=N_EPOCHS, seed=seed)
    vae_all_mse = vae_mse_batch(model, X_all_scaled)
    thr = float(np.percentile(vae_all_mse[val_nonran_idx], FPR_PCT))
    val_nonran_mean_mse = float(vae_all_mse[val_nonran_idx].mean())

    verdicts_var = [binary_verdict(i, thr, vae_all_mse, pmap2, smap2) for i in variant_idx]
    verdicts_c06 = [binary_verdict(i, thr, vae_all_mse, pmap2, smap2) for i in c06_idx]
    verdicts_ben = [binary_verdict(i, thr, vae_all_mse, pmap2, smap2) for i in val_ben_idx]
    verdicts_le  = [binary_verdict(i, thr, vae_all_mse, pmap2, smap2) for i in val_lenc_idx]

    recall     = sum(v == "Attack" for v in verdicts_var) / len(variant_idx)
    c06_recall = sum(v == "Attack" for v in verdicts_c06) / len(c06_idx)
    fpr_ben    = sum(v == "Attack" for v in verdicts_ben) / len(val_ben_idx)
    fpr_lenc   = sum(v == "Attack" for v in verdicts_le) / len(val_lenc_idx)
    fpr_nr     = (sum(v == "Attack" for v in verdicts_ben) + sum(v == "Attack" for v in verdicts_le)) / len(val_nonran_idx)

    per_variant = {}
    for vname in VARIANT_CLASSES:
        vi = [int(i) for i in variant_idx if all_labels[i] == vname]
        n_att = sum(1 for i in vi if binary_verdict(i, thr, vae_all_mse, pmap2, smap2) == "Attack")
        per_variant[vname] = {"n": len(vi), "attack": n_att, "recall": n_att / len(vi)}

    c06_detail = []
    for i in c06_idx:
        mse = float(vae_all_mse[i]); fires = mse > thr
        p1 = float(pmap2[i][1]); stream = float(smap2.get(i, 0.0))
        verdict = binary_verdict(i, thr, vae_all_mse, pmap2, smap2)
        c06_detail.append({"idx": i, "mse": mse, "fires": fires, "cnn_p1": p1, "stream": stream, "verdict": verdict})

    seed_metrics.append({
        "seed": seed, "thr": thr, "val_nonran_mean_mse": val_nonran_mean_mse,
        "recall": recall, "c06_recall": c06_recall,
        "fpr_ben": fpr_ben, "fpr_lenc": fpr_lenc, "fpr_nr": fpr_nr,
        "per_variant": per_variant, "c06_detail": c06_detail,
    })
    seed_models[seed] = (model, vae_all_mse, thr, val_nonran_mean_mse)

    log(f"    seed={seed}  thr={thr:.4f}  val_nonran_mean_mse={val_nonran_mean_mse:.4f}  "
        f"recall={recall:.0%}  C06={c06_recall:.0%}  FPR_Ben={fpr_ben:.1%}  FPR_LEnc={fpr_lenc:.1%}  "
        f"FPR_NR={fpr_nr:.1%}  ({time.time()-t0:.0f}s)")

def summarize(metric_list, key):
    vals = np.array([m[key] for m in metric_list])
    return float(vals.mean()), float(vals.std())

rec_m, rec_s = summarize(seed_metrics, "recall")
c06_m, c06_s = summarize(seed_metrics, "c06_recall")
fb_m,  fb_s  = summarize(seed_metrics, "fpr_ben")
fl_m,  fl_s  = summarize(seed_metrics, "fpr_lenc")
fn_m,  fn_s  = summarize(seed_metrics, "fpr_nr")

log(f"\n  {'='*72}")
log("  MAIN RUN SUMMARY (mean ± std over 3 independent seeds)")
log(f"  {'='*72}")
log(f"  VAE-nonran-only:  Recall={rec_m:.0%}±{rec_s:.0%}  C06={c06_m:.0%}±{c06_s:.0%}  "
    f"FPR_Ben={fb_m:.1%}±{fb_s:.1%}  FPR_LEnc={fl_m:.1%}±{fl_s:.1%}  FPR_NR={fn_m:.1%}±{fn_s:.1%}")
log(f"\n  Reference:  VAE-all-classes (deployed) = 89% / 30% C06 / 5.0% FPR_NR / 70% LOFO")
log(f"              BiLSTM (supervised)         = 99% / 90% C06 / 5.0% FPR_NR / 100% LOFO")
log(f"              TS-TCC-unsupervised          = 93%±2% / 50%±14% C06 / 5.0% FPR_NR / 85% LOFO")

json_out["main_run"] = seed_metrics

# ── Save the best-by-val seed as the canonical deployable artifact ──
best_seed = min(seed_models, key=lambda s: seed_models[s][3])   # lowest val-nonran mean MSE = best fit
best_model, best_mse_all, best_thr, best_val_mse = seed_models[best_seed]
log(f"\n  Canonical (best-by-val, seed={best_seed}) saved as vae_nonran_*")
torch.save(best_model.state_dict(), FEATURES_DIR / "vae_nonran_model.pt")
np.save(FEATURES_DIR / "vae_nonran_scaler_mean.npy",  vae_scaler.mean_)
np.save(FEATURES_DIR / "vae_nonran_scaler_scale.npy", vae_scaler.scale_)
np.save(FEATURES_DIR / "vae_nonran_threshold.npy",    np.array([best_thr]))
np.save(FEATURES_DIR / "vae_nonran_window_scores.npy", best_mse_all)
log("  Saved: vae_nonran_model.pt + scaler + threshold + window_scores  "
    "(vae_fixed_* untouched — all-classes VAE still available for comparison)")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — 6-Family LOFO (VAE fixed across folds — see docstring note; CNN retrained per fold)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72)
log("6-FAMILY LOFO — VAE FIXED (canonical best-by-val model; ransomware families were")
log("never in its training data, so holding one out changes nothing for the VAE).")
log("Only the 2-way CNN is retrained per fold (3-seed pick-best).")
log("=" * 72)

X_tr_text = text_seqs[train_idx]

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

@torch.no_grad()
def probs_and_stream_batch(model, idx_list):
    model.eval()
    pmap, smap = {}, {}
    for i in idx_list:
        i = int(i)
        pmap[i] = probs_2way(model, text_seqs, i)
        smap[i] = 0.0 if pmap[i].argmax() == 1 else streaming_max_2way(model, text_seqs, i)
    return pmap, smap

lofo_results = {}
for family, members in LOFO_FAMILIES.items():
    t0 = time.time()
    log(f"\n  === LOFO: hold out {family} ===")
    held_mask   = np.array([all_labels[i] in members for i in train_idx])
    held_global = train_idx[np.where(held_mask)[0]]
    fold_tr_idx = train_idx[np.where(~held_mask)[0]]
    fold_tr_y2  = train_y_bin[np.where(~held_mask)[0]]

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

    pmap_held, smap_held = probs_and_stream_batch(best_clf, [int(i) for i in held_global])
    pmap_nr,   smap_nr   = probs_and_stream_batch(best_clf, [int(i) for i in val_nonran_idx])

    n_att = sum(1 for i in held_global if binary_verdict(i, best_thr, best_mse_all, pmap_held, smap_held) == "Attack")
    b_fp  = sum(1 for i in val_ben_idx  if binary_verdict(i, best_thr, best_mse_all, pmap_nr, smap_nr) == "Attack")
    le_fp = sum(1 for i in val_lenc_idx if binary_verdict(i, best_thr, best_mse_all, pmap_nr, smap_nr) == "Attack")

    n_held = len(held_global)
    recall = n_att / n_held
    log(f"    N={n_held}  Attack={n_att}  Recall={recall:.0%}  "
        f"FPR_Ben={b_fp/len(val_ben_idx):.1%}  FPR_LEnc={le_fp/len(val_lenc_idx):.1%}  ({time.time()-t0:.0f}s)")

    lofo_results[family] = {
        "n": n_held, "attack": n_att, "recall": recall,
        "fpr_ben": b_fp / len(val_ben_idx), "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": (b_fp + le_fp) / len(val_nonran_idx),
    }
    del best_clf; torch.cuda.empty_cache()

json_out["lofo"] = lofo_results

log(f"\n  LOFO SUMMARY (VAE fixed, canonical seed={best_seed})")
log(f"  {'Family':<12}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
log("  " + "─" * 54)
recalls = []
for fam, r in lofo_results.items():
    log(f"  {fam:<12}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  "
        f"{r['fpr_lenc']:>10.1%}  {r['fpr_nr']:>8.1%}")
    recalls.append(r["recall"])
log(f"  {'MEAN':<12}  {np.mean(recalls):>8.0%}")
log(f"\n  Reference — BiLSTM LOFO: 100%. TS-TCC-unsupervised LOFO: 85-86%. VAE-all-classes LOFO: 70%.")


# ════════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════════
out_txt  = FEATURES_DIR / "vae_nonran_only_results.txt"
out_json = FEATURES_DIR / "vae_nonran_only_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("vae_nonran_only.py done.")
