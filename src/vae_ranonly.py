"""
vae_ranonly.py — VAE Retrained on RANSOMWARE ONLY (inverse formulation) + Likelihood Ratio
==================================================================================================
Per HANDOFF item 55, TASK B + TASK C (August 10, 2026).

Item 54's non-ransomware-only VAE came back a NULL result (reproduced the
all-classes VAE almost exactly). Working hypothesis: reconstruction error in
this feature space is largely INDEPENDENT of what the VAE was trained on — a
16-d bottleneck over 300x23 sequences may capture coarse global structure
rather than familiarity with a specific class distribution. This script is
the extreme test: train the SAME architecture on RANSOMWARE ONLY (the
inverse of item 54) and see if it also lands near 89%/30%/70%.

TASK B — ransomware-only VAE:
  Train the VAE and its StandardScaler on the 900 RANSOMWARE training
  executions only (WannaCry, Ryuk, REvil, LockBit, Darkside, Conti — the 6
  known families within the training pool; validation/held-out variants
  excluded, as always). 3 independent seeds, mean +- std.

  *** SCORE DIRECTION INVERTS. *** This model reconstructs ransomware WELL,
  so it should fire on LOW reconstruction error, not high. Threshold =
  5th percentile of VALIDATION NON-RANSOMWARE MSE; fires when score <
  threshold. This preserves the same ~5% FPR budget as every other head.
  Implementing this backwards would silently invert every result.

  *** REPORTING RULE. *** Recall on the KNOWN ransomware families (e.g. val
  Ransomware) is NOT reported as a result — the model trained on exactly
  that distribution, so a high number there is circular (train-on-test).
  Only Conti_01-07 (blind, never in any training pool) and 6-family LOFO
  count as recall evidence. Known-family scores are computed for diagnostic
  visibility only and explicitly labeled as such in the output.

  LOFO here MUST retrain the VAE per fold (unlike item 54's non-ran-only
  VAE, which never saw ransomware at all): the held-out family IS this
  model's training data, so both the VAE and its scaler are refit on the
  remaining 5 families each fold (single seed=0, same compute trade-off
  documented in the earlier TS-TCC LOFO scripts). The CNN is also retrained
  per fold (3-seed pick-best, same procedure throughout this project).

TASK C — likelihood ratio (bonus, uses both models):
  score = MSE(vae_nonran_model) - MSE(vae_ranonly_model). High score means
  "poorly explained by normal AND well explained by ransomware" — the
  standard likelihood-ratio construction for two-class generative OOD
  detection. Standard (non-inverted) direction: fires when score > 95th
  pct of validation non-ransomware LR score.

★  CORE RULE: VAE trains on the stated subset only; Conti_06 and ALL
   held-out variants NEVER in any training; threshold from validation
   non-ransomware only; blind eval on ALL variants + LOFO; FPR always
   reported. No tuning to Conti_06.
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
FPR_PCT_HI    = 95.0   # standard-direction threshold pct (LR score)
FPR_PCT_LO    = 5.0    # inverted-direction threshold pct (ranonly VAE score)
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
train_ran_mask   = (train_y_bin == 1)
train_ran_global = train_idx[train_ran_mask]   # the ONLY data this VAE sees (900 execs)

val_nonran_idx = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ran_idx    = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])
val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
c06_idx        = [int(i) for i in variant_idx if all_labels[i] == "Conti_06"]

log(f"  Training pool: {len(train_idx)} (ran={train_y_bin.sum()}, non-ran={(train_y_bin==0).sum()})")
log(f"  VAE training set (RANSOMWARE ONLY): {len(train_ran_global)} executions")
log(f"  Val: {len(val_idx)}  Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}, Ran={len(val_ran_idx)}")
log(f"  Variants (blind, never trained on): {len(variant_idx)},  Conti_06: {len(c06_idx)}")

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
# SECTION 3 — VAE (identical architecture to d3_fixed.py / vae_nonran_only.py)
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
# SECTION 4 — Fit scaler on RANSOMWARE TRAINING SUBSET ONLY
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Fitting StandardScaler on ransomware training subset ONLY"); log("=" * 72)

vae_scaler = StandardScaler()
X_ran_raw = feat_23[train_ran_global]          # (900, 300, 23)
vae_scaler.fit(X_ran_raw.reshape(-1, 23))
log(f"  Scaler fit on {len(train_ran_global)} ransomware executions "
    f"(WannaCry/Ryuk/REvil/LockBit/Darkside/Conti) — NOT the full 1800-exec pool.")

def scale_seqs(seqs):
    N, T, C = seqs.shape
    return (vae_scaler.transform(seqs.reshape(-1, C)).astype(np.float32)
            .reshape(N, T, C))

X_ran_scaled = scale_seqs(X_ran_raw)
X_all_scaled = scale_seqs(feat_23)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Fusion verdict (INVERTED — fires on LOW score)
# ════════════════════════════════════════════════════════════════════════════════

def binary_verdict_inverted(i, thr, scores, pmap, smap):
    """Ransomware-only VAE: LOW reconstruction error = ransomware-like."""
    i = int(i)
    p = pmap[i]; s = smap.get(i, 0.0)
    fires = scores[i] < thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"

def binary_verdict_standard(i, thr, scores, pmap, smap):
    """Standard direction (used for the Task C likelihood-ratio score)."""
    i = int(i)
    p = pmap[i]; s = smap.get(i, 0.0)
    fires = scores[i] > thr
    if p[1] >= 0.5: return "Attack"
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN RUN: 3 independent seeds
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("MAIN RUN — VAE trained on RANSOMWARE ONLY, 3 independent seeds"); log("=" * 72)
log("  *** Score direction INVERTED: fires when MSE < 5th-pct-val-non-ran threshold ***")

seed_metrics = []
seed_models  = {}

for seed in SEEDS:
    t0 = time.time()
    model = train_vae(X_ran_scaled, n_epochs=N_EPOCHS, seed=seed)
    vae_all_mse = vae_mse_batch(model, X_all_scaled)
    thr = float(np.percentile(vae_all_mse[val_nonran_idx], FPR_PCT_LO))   # 5th pct, LOW tail
    val_ran_mean_mse = float(vae_all_mse[val_ran_idx].mean())             # "best-by-val" selection metric

    verdicts_var = [binary_verdict_inverted(i, thr, vae_all_mse, pmap2, smap2) for i in variant_idx]
    verdicts_c06 = [binary_verdict_inverted(i, thr, vae_all_mse, pmap2, smap2) for i in c06_idx]
    verdicts_ben = [binary_verdict_inverted(i, thr, vae_all_mse, pmap2, smap2) for i in val_ben_idx]
    verdicts_le  = [binary_verdict_inverted(i, thr, vae_all_mse, pmap2, smap2) for i in val_lenc_idx]
    # Diagnostic only -- NOT a result (train-on-test, model trained on exactly this distribution)
    verdicts_knownran = [binary_verdict_inverted(i, thr, vae_all_mse, pmap2, smap2) for i in val_ran_idx]

    recall     = sum(v == "Attack" for v in verdicts_var) / len(variant_idx)
    c06_recall = sum(v == "Attack" for v in verdicts_c06) / len(c06_idx)
    fpr_ben    = sum(v == "Attack" for v in verdicts_ben) / len(val_ben_idx)
    fpr_lenc   = sum(v == "Attack" for v in verdicts_le) / len(val_lenc_idx)
    fpr_nr     = (sum(v == "Attack" for v in verdicts_ben) + sum(v == "Attack" for v in verdicts_le)) / len(val_nonran_idx)
    diagnostic_knownran_recall = sum(v == "Attack" for v in verdicts_knownran) / len(val_ran_idx)

    per_variant = {}
    for vname in VARIANT_CLASSES:
        vi = [int(i) for i in variant_idx if all_labels[i] == vname]
        n_att = sum(1 for i in vi if binary_verdict_inverted(i, thr, vae_all_mse, pmap2, smap2) == "Attack")
        per_variant[vname] = {"n": len(vi), "attack": n_att, "recall": n_att / len(vi)}

    seed_metrics.append({
        "seed": seed, "thr": thr, "val_ran_mean_mse": val_ran_mean_mse,
        "recall": recall, "c06_recall": c06_recall,
        "fpr_ben": fpr_ben, "fpr_lenc": fpr_lenc, "fpr_nr": fpr_nr,
        "diagnostic_ONLY_known_ransomware_recall": diagnostic_knownran_recall,
        "per_variant": per_variant,
    })
    seed_models[seed] = (model, vae_all_mse, thr, val_ran_mean_mse)

    log(f"    seed={seed}  thr={thr:.4f}  val_ran_mean_mse={val_ran_mean_mse:.4f}  "
        f"recall(Conti_01-07)={recall:.0%}  C06={c06_recall:.0%}  FPR_Ben={fpr_ben:.1%}  "
        f"FPR_LEnc={fpr_lenc:.1%}  FPR_NR={fpr_nr:.1%}  "
        f"[diagnostic-only known-ran recall={diagnostic_knownran_recall:.0%}]  ({time.time()-t0:.0f}s)")

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
log("  Recall/C06/FPR below are the REPORTABLE numbers (Conti_01-07 blind + FPR).")
log("  Known-ransomware-family recall is diagnostic-only and excluded here (see per-seed log above).")
log(f"  {'='*72}")
log(f"  VAE-ransomware-only:  Recall={rec_m:.0%}±{rec_s:.0%}  C06={c06_m:.0%}±{c06_s:.0%}  "
    f"FPR_Ben={fb_m:.1%}±{fb_s:.1%}  FPR_LEnc={fl_m:.1%}±{fl_s:.1%}  FPR_NR={fn_m:.1%}±{fn_s:.1%}")
log(f"\n  Reference:  VAE-all-classes         = 89% / 30% C06 / 5.0% FPR_NR / 70% LOFO")
log(f"              VAE-non-ransomware-only  = 89% / 30% C06 / 5.0% FPR_NR / 70% LOFO")
log(f"              TS-TCC-unsupervised      = 93%±2% / 50%±14% C06 / 5.0% FPR_NR / 85% LOFO")
log(f"              BiLSTM (supervised)      = 99% / 90% C06 / 5.0% FPR_NR / 100% LOFO")

json_out["main_run"] = seed_metrics

# ── Save the best-by-val seed (lowest val-ransomware mean MSE = best fit to its own target class) ──
best_seed = min(seed_models, key=lambda s: seed_models[s][3])
best_model, best_mse_all, best_thr, best_val_ran_mse = seed_models[best_seed]
log(f"\n  Canonical (best-by-val, seed={best_seed}) saved as vae_ranonly_*")
torch.save(best_model.state_dict(), FEATURES_DIR / "vae_ranonly_model.pt")
np.save(FEATURES_DIR / "vae_ranonly_scaler_mean.npy",  vae_scaler.mean_)
np.save(FEATURES_DIR / "vae_ranonly_scaler_scale.npy", vae_scaler.scale_)
np.save(FEATURES_DIR / "vae_ranonly_threshold.npy",    np.array([best_thr]))
np.save(FEATURES_DIR / "vae_ranonly_window_scores.npy", best_mse_all)
log("  Saved: vae_ranonly_model.pt + scaler + threshold + window_scores  "
    "(vae_fixed_* and vae_nonran_* untouched)")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — 6-Family LOFO (VAE + scaler RETRAINED per fold — its training data changes)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72)
log("6-FAMILY LOFO — VAE + scaler retrained per fold (ransomware families ARE its")
log("training data this time). CNN also retrained per fold (3-seed pick-best).")
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

    # -- Fold VAE + scaler (ransomware-only pool minus the held-out family, single seed=0) --
    fold_ran_mask   = np.array([all_labels[i] in RANSOMWARE_CLASSES for i in fold_tr_idx])
    fold_ran_global = fold_tr_idx[fold_ran_mask]   # this family's ransomware excluded
    fold_scaler = StandardScaler()
    fold_ran_raw = feat_23[fold_ran_global]
    fold_scaler.fit(fold_ran_raw.reshape(-1, 23))

    def fold_scale(seqs, scaler=fold_scaler):
        N, T, C = seqs.shape
        return (scaler.transform(seqs.reshape(-1, C)).astype(np.float32).reshape(N, T, C))

    fold_ran_scaled = fold_scale(fold_ran_raw)
    fold_model = train_vae(fold_ran_scaled, n_epochs=N_EPOCHS, seed=0)
    fold_all_scaled = fold_scale(feat_23)
    fold_mse_all = vae_mse_batch(fold_model, fold_all_scaled)
    fold_thr = float(np.percentile(fold_mse_all[val_nonran_idx], FPR_PCT_LO))

    pmap_held, smap_held = probs_and_stream_batch(best_clf, [int(i) for i in held_global])
    pmap_nr,   smap_nr   = probs_and_stream_batch(best_clf, [int(i) for i in val_nonran_idx])

    n_att = sum(1 for i in held_global if binary_verdict_inverted(i, fold_thr, fold_mse_all, pmap_held, smap_held) == "Attack")
    b_fp  = sum(1 for i in val_ben_idx  if binary_verdict_inverted(i, fold_thr, fold_mse_all, pmap_nr, smap_nr) == "Attack")
    le_fp = sum(1 for i in val_lenc_idx if binary_verdict_inverted(i, fold_thr, fold_mse_all, pmap_nr, smap_nr) == "Attack")

    n_held = len(held_global)
    recall = n_att / n_held
    log(f"    N={n_held}  Attack={n_att}  Recall={recall:.0%}  "
        f"FPR_Ben={b_fp/len(val_ben_idx):.1%}  FPR_LEnc={le_fp/len(val_lenc_idx):.1%}  ({time.time()-t0:.0f}s)")

    lofo_results[family] = {
        "n": n_held, "attack": n_att, "recall": recall,
        "fpr_ben": b_fp / len(val_ben_idx), "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": (b_fp + le_fp) / len(val_nonran_idx),
    }
    del best_clf, fold_model; torch.cuda.empty_cache()

json_out["lofo"] = lofo_results

log(f"\n  LOFO SUMMARY (VAE + CNN both retrained per fold)")
log(f"  {'Family':<12}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
log("  " + "─" * 54)
recalls = []
for fam, r in lofo_results.items():
    log(f"  {fam:<12}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  "
        f"{r['fpr_lenc']:>10.1%}  {r['fpr_nr']:>8.1%}")
    recalls.append(r["recall"])
log(f"  {'MEAN':<12}  {np.mean(recalls):>8.0%}")
log(f"\n  Reference — BiLSTM LOFO: 100%. TS-TCC-unsupervised LOFO: 85-86%. VAE-all-classes/non-ran-only LOFO: 70%.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TASK C: Likelihood-ratio score (nonran MSE - ranonly MSE)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("TASK C — LIKELIHOOD RATIO: MSE(non-ran-only) - MSE(ransomware-only)"); log("=" * 72)

nonran_scores_path = FEATURES_DIR / "vae_nonran_window_scores.npy"
if nonran_scores_path.exists():
    # vae_nonran_window_scores.npy was computed on ITS OWN scaler's scaling of feat_23;
    # it is a saved (1970,) array indexed identically to this script's arrays, so no
    # re-scaling is needed here -- just load and difference against this run's ranonly scores.
    nonran_scores = np.load(nonran_scores_path)
    lr_scores = nonran_scores - best_mse_all   # high = poorly explained by normal, well by ransomware
    lr_thr = float(np.percentile(lr_scores[val_nonran_idx], FPR_PCT_HI))

    verdicts_var = [binary_verdict_standard(i, lr_thr, lr_scores, pmap2, smap2) for i in variant_idx]
    verdicts_c06 = [binary_verdict_standard(i, lr_thr, lr_scores, pmap2, smap2) for i in c06_idx]
    verdicts_ben = [binary_verdict_standard(i, lr_thr, lr_scores, pmap2, smap2) for i in val_ben_idx]
    verdicts_le  = [binary_verdict_standard(i, lr_thr, lr_scores, pmap2, smap2) for i in val_lenc_idx]

    lr_recall     = sum(v == "Attack" for v in verdicts_var) / len(variant_idx)
    lr_c06_recall = sum(v == "Attack" for v in verdicts_c06) / len(c06_idx)
    lr_fpr_ben    = sum(v == "Attack" for v in verdicts_ben) / len(val_ben_idx)
    lr_fpr_lenc   = sum(v == "Attack" for v in verdicts_le) / len(val_lenc_idx)
    lr_fpr_nr     = (sum(v == "Attack" for v in verdicts_ben) + sum(v == "Attack" for v in verdicts_le)) / len(val_nonran_idx)

    log(f"  LR threshold (95th pct val non-ran): {lr_thr:.4f}")
    log(f"  Likelihood-ratio VAE:  Recall={lr_recall:.0%}  C06={lr_c06_recall:.0%}  "
        f"FPR_Ben={lr_fpr_ben:.1%}  FPR_LEnc={lr_fpr_lenc:.1%}  FPR_NR={lr_fpr_nr:.1%}")

    c06_lr_detail = [{"idx": i, "nonran_mse": float(nonran_scores[i]), "ranonly_mse": float(best_mse_all[i]),
                       "lr_score": float(lr_scores[i]), "fires": bool(lr_scores[i] > lr_thr)} for i in c06_idx]
    log(f"\n  Conti_06 LR detail:")
    log(f"  {'Idx':>6}  {'NonRan_MSE':>11}  {'RanOnly_MSE':>12}  {'LR_score':>9}  {'Fires':>6}")
    for d in c06_lr_detail:
        log(f"  {d['idx']:>6}  {d['nonran_mse']:>11.4f}  {d['ranonly_mse']:>12.4f}  "
            f"{d['lr_score']:>9.4f}  {'YES' if d['fires'] else 'no':>6}")

    json_out["task_c_likelihood_ratio"] = {
        "thr": lr_thr, "recall": lr_recall, "c06_recall": lr_c06_recall,
        "fpr_ben": lr_fpr_ben, "fpr_lenc": lr_fpr_lenc, "fpr_nr": lr_fpr_nr,
        "c06_detail": c06_lr_detail,
    }
else:
    log("  vae_nonran_window_scores.npy not found on this node — skipping Task C.")
    log("  (Should exist from the VAE_NONRAN_ONLY job; re-run if missing.)")


# ════════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════════
out_txt  = FEATURES_DIR / "vae_ranonly_results.txt"
out_json = FEATURES_DIR / "vae_ranonly_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("vae_ranonly.py done.")
