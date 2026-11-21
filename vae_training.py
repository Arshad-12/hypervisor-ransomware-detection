"""
RanSMAP VAE Training + CV (Full-Scale, 16c)
============================================
Trains the VAE anomaly head on 23-dim features for the full dataset.
Key question: does FPR drop from the local 82% when trained on ~1,800
executions instead of 96?

Runs:
  1. 5-fold CV × 3 seeds (15 runs) — measures FPR vs detection rate
  2. Final model trained on ALL 1,800 training samples — saved for fusion

Saves:
  vae_model.pt          — final model trained on full training pool
  vae_scaler.npy        — scaler mean/scale for inference
  vae_cv_results.json   — CV numbers
  vae_results.txt       — human-readable summary
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
OUTPUT_DIR   = FEATURES_DIR

# ── Class definitions ─────────────────────────────────────────────────────────
RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS      = 150
LR          = 1e-3
BATCH_SIZE  = 32
BETA_TARGET = 1e-3
WARMUP      = 20
CV_SEEDS    = [42, 123, 7]
K_FOLDS     = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Model ─────────────────────────────────────────────────────────────────────
class TimeSeriesVAE(nn.Module):
    def __init__(self, seq_len=300, n_features=23, latent_dim=16):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        inp = seq_len * n_features   # 6900
        self.enc = nn.Sequential(
            nn.Linear(inp, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 512),        nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, inp),
        )

    def encode(self, x):
        h = self.enc(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(z).view(z.size(0), self.seq_len, self.n_features)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return self.decode(z), mu, logvar


# ── Dataset ───────────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, sequences):
        self.X = torch.tensor(sequences, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


def recon_mse(model, arr):
    """Per-sample reconstruction MSE using deterministic mu."""
    dl = DataLoader(SeqDataset(arr), batch_size=64, shuffle=False)
    errs = []
    model.eval()
    with torch.no_grad():
        for bx in dl:
            bx = bx.to(device)
            mu_b, _ = model.encode(bx)
            rx = model.decode(mu_b)
            errs.append(((rx - bx) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(errs)


def train_vae(sequences_scaled, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m   = TimeSeriesVAE().to(device)
    opt = optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
    sch = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    dl  = DataLoader(SeqDataset(sequences_scaled), batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(1, EPOCHS + 1):
        m.train()
        beta = BETA_TARGET * min(1.0, ep / WARMUP)
        for bx in dl:
            bx = bx.to(device)
            opt.zero_grad()
            recon_b, mu_b, logvar_b = m(bx)
            loss = vae_loss(recon_b, bx, mu_b, logvar_b, beta)
            loss.backward()
            opt.step()
        sch.step()
        if ep % 30 == 0 or ep == 1:
            print(f"    epoch {ep:>3}/{EPOCHS}", flush=True)
    return m


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading feature files...")
meta_df = pd.read_csv(FEATURES_DIR / "metadata.csv")

sequences, labels, groups = [], [], []
for _, row in meta_df.iterrows():
    sequences.append(np.load(row["filepath"]))
    labels.append(row["label"])
    groups.append(row["group"])

sequences = np.array(sequences, dtype=np.float32)
labels    = np.array(labels)
groups    = np.array(groups)
print(f"Loaded: {sequences.shape}")

# Training pool: original + extra, known classes
train_mask   = np.isin(groups, ["original", "extra"]) & np.isin(labels, ALL_ORIGINAL_CLASSES)
variant_mask = np.isin(labels, VARIANT_CLASSES)

train_seqs = sequences[train_mask]
var_seqs   = sequences[variant_mask]
var_labels = labels[variant_mask]

print(f"Training pool: {train_mask.sum()}  |  Variants (held out): {variant_mask.sum()}")
print(f"Local reference: 120 originals → now {train_mask.sum()} (full scale)")


# ── 5-fold CV × 3 seeds ───────────────────────────────────────────────────────
print(f"\nRunning {len(CV_SEEDS)} seeds × {K_FOLDS} folds = {len(CV_SEEDS)*K_FOLDS} CV runs...")

run_results      = []
variant_errs_all = defaultdict(list)
variant_det_95   = defaultdict(list)
variant_det_fx   = defaultdict(list)

for seed in CV_SEEDS:
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_seqs)):
        print(f"\n[seed={seed} fold={fold}]", flush=True)

        # Scale: fit on train fold only
        sc = StandardScaler()
        sc.fit(train_seqs[tr_idx].reshape(-1, 23))
        tr_s  = sc.transform(train_seqs[tr_idx].reshape(-1, 23)).reshape(len(tr_idx),  300, 23).astype(np.float32)
        val_s = sc.transform(train_seqs[val_idx].reshape(-1, 23)).reshape(len(val_idx), 300, 23).astype(np.float32)
        var_s = sc.transform(var_seqs.reshape(-1, 23)).reshape(len(var_seqs),           300, 23).astype(np.float32)

        m = train_vae(tr_s, seed=seed * 100 + fold)

        tr_errs  = recon_mse(m, tr_s)
        val_errs = recon_mse(m, val_s)
        var_errs = recon_mse(m, var_s)

        t95 = np.percentile(tr_errs, 95)
        orig_mean = val_errs.mean()
        var_mean  = var_errs.mean()
        ratio     = var_mean / orig_mean if orig_mean > 0 else float("nan")
        fpr_95    = (val_errs > t95).mean()
        fpr_fx    = (val_errs > 0.5).mean()

        for vi, vlabel in enumerate(var_labels):
            e = var_errs[vi]
            variant_errs_all[vlabel].append(e)
            variant_det_95[vlabel].append(bool(e > t95))
            variant_det_fx[vlabel].append(bool(e > 0.5))

        run_results.append(dict(
            seed=seed, fold=fold,
            orig_mean=orig_mean, var_mean=var_mean, ratio=ratio,
            t95=t95, fpr_95=fpr_95, fpr_fx=fpr_fx,
            det_95=(var_errs > t95).mean(),
            det_fx=(var_errs > 0.5).mean(),
        ))
        print(f"  ratio={ratio:.2f}x | FPR@95={fpr_95:.1%} | det@95={(var_errs > t95).mean():.1%} | t95={t95:.3f}", flush=True)


# ── CV summary ────────────────────────────────────────────────────────────────
df = pd.DataFrame(run_results)

lines = []
lines.append("=" * 70)
lines.append(f"VAE CV — Full Dataset ({train_mask.sum()} training executions)")
lines.append(f"Local reference: 120 executions | FPR@95 local: 81.9%")
lines.append("=" * 70)
lines.append(f"\n{'Metric':<40} {'Mean':>8} {'Std':>8}")
lines.append("-" * 58)

for col, label in [
    ("ratio",    "Variant/Original MSE ratio"),
    ("orig_mean","Original MSE (held-out fold)"),
    ("var_mean", "Variant MSE"),
    ("fpr_95",   "FPR @ 95th pct"),
    ("fpr_fx",   "FPR @ fixed 0.5"),
    ("det_95",   "Detection rate @ 95th pct"),
    ("det_fx",   "Detection rate @ fixed 0.5"),
]:
    lines.append(f"  {label:<38} {df[col].mean():8.4f} {df[col].std():8.4f}")

lines.append("\n--- Per-variant detection @ 95th pct ---")
lines.append(f"{'Variant':<12} {'MSE mean':>10} {'Det@95%':>10} {'Det@0.5':>10}")
lines.append("-" * 44)
for vlabel in VARIANT_CLASSES:
    if vlabel not in variant_errs_all:
        continue
    d95 = np.mean(variant_det_95[vlabel])
    dfx = np.mean(variant_det_fx[vlabel])
    mse = np.mean(variant_errs_all[vlabel])
    flag = " ← watch" if vlabel == "Conti_06" else ""
    lines.append(f"  {vlabel:<10} {mse:>10.4f} {d95:>10.1%} {dfx:>10.1%}{flag}")

lines.append(f"\nLocal FPR@95 was 81.9% on 96 training samples.")
lines.append(f"Full-scale FPR@95:  {df['fpr_95'].mean():.1%} ± {df['fpr_95'].std():.1%}")
lines.append(f"FPR improvement: {'YES ✓' if df['fpr_95'].mean() < 0.5 else 'PARTIAL' if df['fpr_95'].mean() < 0.7 else 'still high — VAE needs more data or regularisation'}")

summary_text = "\n".join(lines)
print("\n" + summary_text)


# ── Final model on full training pool ────────────────────────────────────────
print("\nTraining final model on full training pool...")
sc_final = StandardScaler()
sc_final.fit(train_seqs.reshape(-1, 23))
train_scaled = sc_final.transform(train_seqs.reshape(-1, 23)).reshape(len(train_seqs), 300, 23).astype(np.float32)

final_model = train_vae(train_scaled, seed=42)

# Compute final threshold from training reconstruction errors
train_errs_final = recon_mse(final_model, train_scaled)
t95_final = np.percentile(train_errs_final, 95)
print(f"Final model threshold (95th pct of training): {t95_final:.4f}")

# Variant detection with final model
var_scaled_final = sc_final.transform(var_seqs.reshape(-1, 23)).reshape(len(var_seqs), 300, 23).astype(np.float32)
var_errs_final   = recon_mse(final_model, var_scaled_final)
print("\nFinal model — variant detection:")
for vlabel in VARIANT_CLASSES:
    mask = var_labels == vlabel
    if mask.sum() == 0:
        continue
    det = (var_errs_final[mask] > t95_final).mean()
    print(f"  {vlabel}: {det:.1%} detected  (mean MSE={var_errs_final[mask].mean():.4f})")

# Save final model
torch.save(final_model.state_dict(), OUTPUT_DIR / "vae_model.pt")
np.save(OUTPUT_DIR / "vae_scaler_mean.npy",  sc_final.mean_)
np.save(OUTPUT_DIR / "vae_scaler_scale.npy", sc_final.scale_)
np.save(OUTPUT_DIR / "vae_threshold.npy",    np.array([t95_final]))

# Save CV results
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

cv_summary = {
    "n_training": int(train_mask.sum()),
    "n_variants": int(variant_mask.sum()),
    "runs": run_results,
    "aggregate": {col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
                  for col in ["ratio", "fpr_95", "fpr_fx", "det_95", "det_fx"]},
    "per_variant": {
        vl: {"mse_mean": float(np.mean(variant_errs_all[vl])),
             "det_95": float(np.mean(variant_det_95[vl])),
             "det_fx": float(np.mean(variant_det_fx[vl]))}
        for vl in variant_errs_all
    },
    "final_model_threshold_95": float(t95_final),
}
with open(OUTPUT_DIR / "vae_cv_results.json", "w") as f:
    json.dump(cv_summary, f, indent=2, cls=NpEncoder)

with open(OUTPUT_DIR / "vae_results.txt", "w") as f:
    f.write(summary_text)

print("\nSaved:")
print(f"  {OUTPUT_DIR}/vae_model.pt")
print(f"  {OUTPUT_DIR}/vae_scaler_mean.npy / vae_scaler_scale.npy")
print(f"  {OUTPUT_DIR}/vae_threshold.npy  ({t95_final:.4f})")
print(f"  {OUTPUT_DIR}/vae_cv_results.json")
print(f"  {OUTPUT_DIR}/vae_results.txt")
print("\nDone.")
