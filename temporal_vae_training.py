"""
RanSMAP Temporal VAE Training (Full-Scale, Alliance)
=====================================================
Replaces the flat MLP VAE with a sequence-aware LSTM VAE that processes
the (300, 23) feature matrix window-by-window instead of flattening to 6900-d.

Why: the flat VAE averaged away the entropy trajectory. Conti_06 spikes to
moderate entropy (0.52–0.69) in specific windows but is quiet otherwise — the
mean MSE across 300 windows comes out low (0.24–0.47, below threshold 0.6720).
An LSTM encoder learns the temporal pattern itself; that spike from near-zero to
moderate-then-back IS anomalous relative to all three training classes:
  - Benign (Firefox/Office/Idle): flat quiet throughout
  - Legit-Enc (AESCrypt/Zip/SDelete): uniformly high from early on
  - Ransomware (Conti/WannaCry/...): high from the start, sustained

Trajectory check (June 28 2026): all 8 missed Conti_06 executions spike
within the 30s window → temporal pattern is in the data; model must see it.

Architecture:
  Encoder: 2-layer bidirectional LSTM (23→64 hidden) → last hidden state (256-d)
           → linear to mu (32) and logvar (32)
  Decoder: linear (32→256) → 2-layer LSTM → linear (64→23) per timestep

Training: same split as flat VAE — 1,800 original+extra executions, 5-fold CV × 3 seeds.
Final model trained on full 1,800 for fusion use.

Saves:
  temporal_vae_model.pt
  temporal_vae_scaler_mean.npy / _scale.npy
  temporal_vae_threshold.npy
  temporal_vae_cv_results.json
  temporal_vae_results.txt
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

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
OUTPUT_DIR   = FEATURES_DIR

RANSOMWARE_CLASSES   = ["WannaCry","Ryuk","REvil","LockBit","Darkside","Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt","SDelete","Zip"]
BENIGN_CLASSES       = ["Firefox","Office","Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_DIM  = 64
LATENT_DIM  = 32
NUM_LAYERS  = 2
EPOCHS      = 100
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
class TemporalVAE(nn.Module):
    """
    Bidirectional LSTM encoder + LSTM decoder.
    Input: (batch, seq_len=300, n_features=23)
    """
    def __init__(self, n_features=23, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_features = n_features

        # Bidirectional encoder — last timestep hidden from both directions
        self.encoder_lstm = nn.LSTM(
            n_features, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        # Bi-LSTM gives 2×hidden_dim
        self.fc_mu     = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=0.2
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def encode(self, x):
        # x: (batch, 300, 23)
        _, (h_n, _) = self.encoder_lstm(x)
        # h_n: (num_layers*2, batch, hidden_dim) — take last layer, both directions
        h_fwd = h_n[-2]   # forward,  last layer
        h_bwd = h_n[-1]   # backward, last layer
        h = torch.cat([h_fwd, h_bwd], dim=1)   # (batch, hidden_dim*2)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, seq_len=300):
        # z: (batch, latent_dim)
        h0 = self.fc_decode(z).unsqueeze(0).repeat(self.num_layers, 1, 1)   # (layers, batch, hidden)
        c0 = torch.zeros_like(h0)
        # Repeat z as input at each timestep
        inp = self.fc_decode(z).unsqueeze(1).repeat(1, seq_len, 1)   # (batch, 300, hidden)
        out, _ = self.decoder_lstm(inp, (h0, c0))   # (batch, 300, hidden)
        return self.output_layer(out)   # (batch, 300, 23)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return self.decode(z, x.size(1)), mu, logvar


# ── Dataset + loss ────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, sequences):
        self.X = torch.tensor(sequences, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld        = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


def recon_mse(model, arr):
    """Per-sample reconstruction MSE using deterministic mu (eval mode)."""
    dl   = DataLoader(SeqDataset(arr), batch_size=64, shuffle=False)
    errs = []
    model.eval()
    with torch.no_grad():
        for bx in dl:
            bx  = bx.to(device)
            mu_b, _ = model.encode(bx)
            rx  = model.decode(mu_b, bx.size(1))
            errs.append(((rx - bx) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(errs)


def train_temporal_vae(sequences_scaled, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m   = TemporalVAE().to(device)
    opt = optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
    sch = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
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
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sch.step()
        if ep % 25 == 0 or ep == 1:
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

train_mask   = np.isin(groups, ["original","extra"]) & np.isin(labels, ALL_ORIGINAL_CLASSES)
variant_mask = np.isin(labels, VARIANT_CLASSES)

train_seqs = sequences[train_mask]
var_seqs   = sequences[variant_mask]
var_labels = labels[variant_mask]

print(f"Training pool: {train_mask.sum()}  |  Variants (held out): {variant_mask.sum()}")


# ── 5-fold CV × 3 seeds ───────────────────────────────────────────────────────
print(f"\nRunning {len(CV_SEEDS)} seeds × {K_FOLDS} folds = {len(CV_SEEDS)*K_FOLDS} runs...")

run_results      = []
variant_errs_all = defaultdict(list)
variant_det_95   = defaultdict(list)

for seed in CV_SEEDS:
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_seqs)):
        print(f"\n[seed={seed} fold={fold}]", flush=True)

        sc = StandardScaler()
        sc.fit(train_seqs[tr_idx].reshape(-1, 23))
        tr_s  = sc.transform(train_seqs[tr_idx].reshape(-1, 23)).reshape(len(tr_idx),  300, 23).astype(np.float32)
        val_s = sc.transform(train_seqs[val_idx].reshape(-1, 23)).reshape(len(val_idx), 300, 23).astype(np.float32)
        var_s = sc.transform(var_seqs.reshape(-1, 23)).reshape(len(var_seqs),           300, 23).astype(np.float32)

        m = train_temporal_vae(tr_s, seed=seed * 100 + fold)

        tr_errs  = recon_mse(m, tr_s)
        val_errs = recon_mse(m, val_s)
        var_errs = recon_mse(m, var_s)

        t95     = np.percentile(tr_errs, 95)
        fpr_95  = (val_errs > t95).mean()
        det_95  = (var_errs > t95).mean()
        ratio   = var_errs.mean() / val_errs.mean() if val_errs.mean() > 0 else float("nan")

        for vi, vlabel in enumerate(var_labels):
            variant_errs_all[vlabel].append(var_errs[vi])
            variant_det_95[vlabel].append(bool(var_errs[vi] > t95))

        run_results.append(dict(seed=seed, fold=fold, t95=t95,
                                fpr_95=fpr_95, det_95=det_95, ratio=ratio))
        print(f"  ratio={ratio:.2f}x | FPR@95={fpr_95:.1%} | det@95={det_95:.1%} | t95={t95:.4f}", flush=True)

        # Print Conti_06 specifically each fold (blind — evaluating on all variants)
        c06_mask = var_labels == "Conti_06"
        c06_det  = (var_errs[c06_mask] > t95).mean()
        print(f"  Conti_06 det@95: {c06_det:.1%}  (flat VAE was 0% on missed execs)", flush=True)


# ── CV summary ────────────────────────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame(run_results)

lines = []
lines.append("=" * 70)
lines.append(f"Temporal VAE CV — Full Dataset ({train_mask.sum()} training executions)")
lines.append(f"Flat MLP VAE reference: FPR@95 = 15.3% ± 2.4%")
lines.append("=" * 70)
lines.append(f"\n{'Metric':<40} {'Mean':>8} {'Std':>8}")
lines.append("-" * 58)
for col, label in [
    ("ratio",   "Variant/Original MSE ratio"),
    ("fpr_95",  "FPR @ 95th pct"),
    ("det_95",  "Detection rate @ 95th pct (all variants)"),
]:
    lines.append(f"  {label:<38} {df[col].mean():8.4f} {df[col].std():8.4f}")

lines.append("\n--- Per-variant detection @ 95th pct ---")
lines.append(f"{'Variant':<12} {'MSE mean':>10} {'Det@95%':>10}")
lines.append("-" * 34)
for vlabel in VARIANT_CLASSES:
    if vlabel not in variant_errs_all: continue
    d95 = np.mean(variant_det_95[vlabel])
    mse = np.mean(variant_errs_all[vlabel])
    flag = " ◀" if vlabel == "Conti_06" else ""
    lines.append(f"  {vlabel:<10} {mse:>10.4f} {d95:>10.1%}{flag}")

lines.append(f"\nFlat VAE FPR@95: 15.3%  |  Temporal VAE FPR@95: {df['fpr_95'].mean():.1%}")
lines.append(f"Key question: does Temporal VAE catch Conti_06 that flat VAE missed?")

summary_text = "\n".join(lines)
print("\n" + summary_text)


# ── Final model on full training pool ────────────────────────────────────────
print("\nTraining final model on full 1,800 training pool...")
sc_final = StandardScaler()
sc_final.fit(train_seqs.reshape(-1, 23))
train_scaled = sc_final.transform(train_seqs.reshape(-1, 23)).reshape(len(train_seqs), 300, 23).astype(np.float32)

final_model = train_temporal_vae(train_scaled, seed=42)

train_errs_final = recon_mse(final_model, train_scaled)
t95_final = np.percentile(train_errs_final, 95)
print(f"Final threshold (95th pct): {t95_final:.4f}")

var_scaled_final = sc_final.transform(var_seqs.reshape(-1, 23)).reshape(len(var_seqs), 300, 23).astype(np.float32)
var_errs_final   = recon_mse(final_model, var_scaled_final)

print("\nFinal model — per-variant detection (blind evaluation):")
for vlabel in VARIANT_CLASSES:
    mask = var_labels == vlabel
    if mask.sum() == 0: continue
    det = (var_errs_final[mask] > t95_final).mean()
    mse = var_errs_final[mask].mean()
    print(f"  {vlabel}: {det:.1%} detected  (mean MSE={mse:.4f})")

# Save
torch.save(final_model.state_dict(), OUTPUT_DIR / "temporal_vae_model.pt")
np.save(OUTPUT_DIR / "temporal_vae_scaler_mean.npy",  sc_final.mean_)
np.save(OUTPUT_DIR / "temporal_vae_scaler_scale.npy", sc_final.scale_)
np.save(OUTPUT_DIR / "temporal_vae_threshold.npy",    np.array([t95_final]))

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

cv_out = {
    "n_training": int(train_mask.sum()),
    "architecture": "BiLSTM encoder + LSTM decoder",
    "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM,
    "runs": run_results,
    "aggregate": {col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
                  for col in ["ratio","fpr_95","det_95"]},
    "per_variant": {
        vl: {"mse_mean": float(np.mean(variant_errs_all[vl])),
             "det_95":   float(np.mean(variant_det_95[vl]))}
        for vl in variant_errs_all
    },
    "final_threshold": float(t95_final),
}
with open(OUTPUT_DIR / "temporal_vae_cv_results.json","w") as f:
    json.dump(cv_out, f, indent=2, cls=NpEncoder)
with open(OUTPUT_DIR / "temporal_vae_results.txt","w") as f:
    f.write(summary_text)

print(f"\nSaved: temporal_vae_model.pt / threshold {t95_final:.4f}")
print("Done.")
