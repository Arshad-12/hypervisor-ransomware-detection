"""
RanSMAP mpnet-Alone Classifier (Priority 1)
=============================================
Trains the same 1D-CNN classifier as classifier_training.py but using
mpnet (all-mpnet-base-v2, 768-d) embeddings instead of MiniLM (384-d).

This is the "text-embedding-only ablation": replaces the MiniLM primary
classifier with an mpnet-based one to test whether a stronger LM improves
3-way separation and zero-day detection.

mpnet_sequences.npy must already exist (written by diag_d2b_mpnet_embed.py).

Saves:
  mpnet_classifier.pt            — best checkpoint
  cnn_mpnet_penultimate.npy      — 64-d pooled embeddings for all 1970 execs
  mpnet_fewshot_split.npz        — train/val/variant splits (same 80/20 seed)
  mpnet_classifier_results.txt   — val report + zero-day summary + fusion eval
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
OUTPUT_DIR   = FEATURES_DIR

# ── Class definitions (same as classifier_training.py) ────────────────────────
RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]

# ── Hyperparameters ───────────────────────────────────────────────────────────
EMB_DIM    = 768   # mpnet output dim (vs 384 for MiniLM)
EPOCHS     = 80
LR         = 1e-3
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Model (identical architecture, only emb_dim changes) ──────────────────────
class TextEmbeddingClassifier(nn.Module):
    def __init__(self, seq_len=300, emb_dim=EMB_DIM, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_classes),
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        x_t    = x.permute(0, 2, 1)       # (batch, emb_dim, 300)
        feat   = self.conv(x_t)            # (batch, 64, 300)
        pooled = feat.mean(dim=2)          # (batch, 64)
        logits = self.classifier(pooled)
        recon  = self.reconstructor(pooled)
        return logits, recon, pooled


# ── Dataset ───────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels,    dtype=torch.long)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ── Load mpnet embeddings ─────────────────────────────────────────────────────
print("Loading mpnet embeddings...")
mpnet_seqs = np.load(FEATURES_DIR / "mpnet_sequences.npy")   # (1970, 300, 768)
all_labels = np.load(FEATURES_DIR / "text_labels.npy")
all_groups = np.load(FEATURES_DIR / "text_groups.npy")
print(f"Loaded mpnet_sequences: {mpnet_seqs.shape}")

def get_3way(label):
    if label in RANSOMWARE_CLASSES: return "ransomware"
    if label in LEGIT_ENC_CLASSES:  return "legit_encryption"
    if label in BENIGN_CLASSES:     return "benign"
    return None

all_3way = np.array([get_3way(l) for l in all_labels])

train_pool_mask = (
    np.isin(all_groups, ["original", "extra"]) &
    np.isin(all_labels, ALL_ORIGINAL_CLASSES)
)
variant_mask = np.isin(all_labels, VARIANT_CLASSES)
mix_mask     = all_groups == "mix"

print(f"\nData split:")
print(f"  Train pool (original+extra): {train_pool_mask.sum()}")
print(f"  Variant (zero-day):          {variant_mask.sum()}")
print(f"  Mix (simultaneous):          {mix_mask.sum()}")

# Encode 3-way labels — same as classifier_training.py
le           = LabelEncoder()
pool_3way    = all_3way[train_pool_mask]
pool_labels  = le.fit_transform(pool_3way)
print(f"\nLabel encoding: { {c: int(le.transform([c])[0]) for c in le.classes_} }")

pool_indices = np.where(train_pool_mask)[0]
train_idx, val_idx = train_test_split(
    np.arange(len(pool_indices)), test_size=0.2,
    random_state=42, stratify=pool_labels
)
train_global = pool_indices[train_idx]
val_global   = pool_indices[val_idx]

print(f"  Train: {len(train_global)}  Val: {len(val_global)}")

train_ds     = SequenceDataset(mpnet_seqs[train_global], pool_labels[train_idx])
val_ds       = SequenceDataset(mpnet_seqs[val_global],   pool_labels[val_idx])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ── Train ─────────────────────────────────────────────────────────────────────
model     = TextEmbeddingClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val Acc':>9}")
print("-" * 45)

best_val_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward(); optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, _, _ = model(batch_x)
            val_loss += criterion(logits, batch_y).item()
            correct  += (logits.argmax(1) == batch_y).sum().item()
            total    += len(batch_y)

    avg_train = epoch_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    acc       = correct / total
    scheduler.step()

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), OUTPUT_DIR / "mpnet_classifier.pt")

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6}  {avg_train:>11.4f}  {avg_val:>10.4f}  {acc:>8.2%}")

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

model.load_state_dict(torch.load(OUTPUT_DIR / "mpnet_classifier.pt", map_location=device))
model.eval()


# ── Validation report ─────────────────────────────────────────────────────────
all_preds, all_true = [], []
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        logits, _, _ = model(batch_x)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_true.extend(batch_y.numpy())

report = classification_report(all_true, all_preds, target_names=le.classes_)
print("\nValidation Report:")
print(report)
results_lines = [report]


# ── Extract penultimate embeddings for ALL 1970 executions ────────────────────
print("Extracting 64-d penultimate embeddings for all 1970 executions...")
all_ds      = DataLoader(
    SequenceDataset(mpnet_seqs, np.zeros(len(mpnet_seqs), dtype=np.int64)),
    batch_size=64, shuffle=False, num_workers=2
)
penultimate = []
all_probs   = []   # (1970, 3) softmax probabilities — used for fusion
with torch.no_grad():
    for batch_x, _ in all_ds:
        batch_x = batch_x.to(device)
        logits, _, pooled = model(batch_x)
        penultimate.append(pooled.cpu().numpy())
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

penultimate = np.concatenate(penultimate, axis=0)   # (1970, 64)
all_probs   = np.concatenate(all_probs,   axis=0)   # (1970, 3)
print(f"Penultimate shape: {penultimate.shape}")
np.save(OUTPUT_DIR / "cnn_mpnet_penultimate.npy", penultimate)


# ── Zero-day variant evaluation (oracle-style per exec) ───────────────────────
RAN_IDX = list(le.classes_).index("ransomware")

print("\nZero-Day Detection — Conti Variants (mpnet classifier)")
print("=" * 65)
print(f"{'Exec':>6}  {'Label':<12}  {'Pred Class':<20}  {'P(ran)':>8}  {'Correct?':>9}")
print("-" * 65)

results_lines.append("\nZero-Day Detection — Conti Variants (mpnet classifier)")
results_lines.append("=" * 65)

variant_correct = 0
variant_total   = 0
for variant in VARIANT_CLASSES:
    mask = all_labels == variant
    if mask.sum() == 0:
        continue
    var_probs = all_probs[mask]   # (n_exec, 3) — already extracted
    preds     = var_probs.argmax(axis=1)
    for exec_i, (pred, probs) in enumerate(zip(preds, var_probs)):
        pred_cls  = le.classes_[pred]
        is_ran    = pred == RAN_IDX
        variant_correct += int(is_ran)
        variant_total   += 1
        line = f"  {exec_i:>4}  {variant:<12}  {pred_cls:<20}  {probs[RAN_IDX]:>8.4f}  {'HIT' if is_ran else 'miss':>9}"
        print(line)
        results_lines.append(line)

recall_line = f"\nVariant recall: {variant_correct}/{variant_total} = {variant_correct/max(1,variant_total):.1%}"
print(recall_line)
results_lines.append(recall_line)


# ── Fusion eval using mpnet primary probs ─────────────────────────────────────
# Load VAE stream scores and other artefacts needed for fusion_v4 logic
print("\n" + "="*65)
print("Fusion eval: mpnet primary + VAE stream + flat MSE anomaly")
print("="*65)

try:
    from numpy.lib.npyio import NpzFile
    fewshot_data = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
    vae_scores   = np.load(FEATURES_DIR / "vae_window_scores.npy")   # (1970, 300)
    vae_thresh   = float(np.load(FEATURES_DIR / "vae_threshold.npy"))
    mix_mask_    = fewshot_data["mix_mask"]

    # Recreate val / train / variant masks using the same split as classifier_training.py
    # (fewshot_split_cnn.npz was written by classifier_training.py with random_state=42)
    val_idx_global   = fewshot_data["val_idx"]
    variant_mask_    = fewshot_data["variant_mask"]
    orig_labels      = fewshot_data["all_labels"]
    orig_groups      = fewshot_data["all_groups"]

    STREAM_THRESH = vae_thresh

    RAN_IDX_   = RAN_IDX
    LENC_IDX_  = list(le.classes_).index("legit_encryption")
    BENIGN_IDX_= list(le.classes_).index("benign")

    def stream_score(vae_row):
        """Fraction of windows above VAE threshold."""
        return float((vae_row > vae_thresh).mean())

    def fuse_v4_mpnet(p, s):
        """Same logic as fuse_v4 but using mpnet primary probs."""
        if p[RAN_IDX_] >= p[BENIGN_IDX_] and p[RAN_IDX_] >= p[LENC_IDX_]: return "Attack"
        if p[LENC_IDX_] >= p[BENIGN_IDX_]:
            if s >= STREAM_THRESH: return "Attack"
            return "Benign"
        if s >= STREAM_THRESH: return "Attack"
        return "Benign"

    def run_eval(mask, name):
        idxs   = np.where(mask)[0]
        labels = orig_labels[idxs]
        preds  = []
        for idx in idxs:
            p = all_probs[idx]
            s = stream_score(vae_scores[idx])
            preds.append(fuse_v4_mpnet(p, s))
        preds  = np.array(preds)
        # Ransomware recall (binary: ran vs rest)
        ran_mask  = np.array([l in RANSOMWARE_CLASSES or l in VARIANT_CLASSES for l in labels])
        ben_mask  = ~ran_mask
        recall    = (preds[ran_mask]  == "Attack").mean() if ran_mask.any() else float("nan")
        fpr       = (preds[ben_mask]  == "Attack").mean() if ben_mask.any() else float("nan")
        n_ran     = ran_mask.sum()
        n_ben     = ben_mask.sum()
        print(f"\n{name} (n_ran={n_ran}, n_ben={n_ben})")
        print(f"  Recall: {recall:.1%}   FPR: {fpr:.2%}")
        return recall, fpr

    RANSOMWARE_CLASSES_SET = set(RANSOMWARE_CLASSES)

    # Val benign FPR
    val_mask   = np.zeros(len(orig_labels), dtype=bool)
    val_mask[val_idx_global] = True
    run_eval(val_mask & np.array([l in BENIGN_CLASSES + LEGIT_ENC_CLASSES
                                  for l in orig_labels]), "Val non-ransomware FPR")

    # Variant recall (all 7 Conti variants)
    run_eval(variant_mask_, "Variant (Conti_01-07) recall")

    print("\n(Compare to fusion_v4 MiniLM baseline: 84% variant recall / 1.7% FPR)")

except FileNotFoundError as e:
    print(f"Skipping fusion eval — missing file: {e}")
    print("Run fusion_v4.py first to generate vae_window_scores.npy and vae_threshold.npy")


# ── Save everything ───────────────────────────────────────────────────────────
np.savez(OUTPUT_DIR / "mpnet_fewshot_split.npz",
         train_idx=train_global,
         val_idx=val_global,
         variant_mask=variant_mask,
         mix_mask=mix_mask,
         all_labels=all_labels,
         all_groups=all_groups,
         le_classes=le.classes_)

np.save(OUTPUT_DIR / "mpnet_all_probs.npy", all_probs)   # (1970, 3) — for combined fusion later

with open(OUTPUT_DIR / "mpnet_classifier_results.txt", "w") as f:
    f.write("\n".join(results_lines))

print(f"\nSaved:")
print(f"  {OUTPUT_DIR}/mpnet_classifier.pt")
print(f"  {OUTPUT_DIR}/cnn_mpnet_penultimate.npy  {penultimate.shape}")
print(f"  {OUTPUT_DIR}/mpnet_all_probs.npy         {all_probs.shape}")
print(f"  {OUTPUT_DIR}/mpnet_fewshot_split.npz")
print(f"  {OUTPUT_DIR}/mpnet_classifier_results.txt")
print("\nDone.")
