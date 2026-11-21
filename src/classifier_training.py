"""
RanSMAP Classifier Training (Full-Scale)
=========================================
Trains the 1D-CNN 3-way classifier on the full ~1,440 original+extra executions.
Holds out Conti variants (Conti_01–07) entirely as zero-day test set.
Saves:
  classifier.pt              — trained model weights
  cnn_penultimate.npy        — 64-d penultimate embeddings for all executions
  fewshot_split_cnn.npz      — train/val/variant splits for downstream use
  classifier_results.txt     — val report + zero-day summary
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

# ── Class definitions ─────────────────────────────────────────────────────────
RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS     = 80
LR         = 1e-3
BATCH_SIZE = 32   # larger than local (8) — H100 handles it fine
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Model ─────────────────────────────────────────────────────────────────────
class TextEmbeddingClassifier(nn.Module):
    def __init__(self, seq_len=300, emb_dim=384, n_classes=3):
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
        x_t    = x.permute(0, 2, 1)       # (batch, 384, 300)
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading embeddings...")
text_seqs  = np.load(FEATURES_DIR / "text_sequences.npy")   # (1970, 300, 384)
all_labels = np.load(FEATURES_DIR / "text_labels.npy")
all_groups = np.load(FEATURES_DIR / "text_groups.npy")
print(f"Loaded: {text_seqs.shape}")

# Reconstruct proper 3-way labels from class names
# (text_3way.npy labels extra/mix/variant all as "variant" — not useful here)
def get_3way(label):
    if label in RANSOMWARE_CLASSES: return "ransomware"
    if label in LEGIT_ENC_CLASSES:  return "legit_encryption"
    if label in BENIGN_CLASSES:     return "benign"
    return None   # variant or mix

all_3way = np.array([get_3way(l) for l in all_labels])

# Training pool: original + extra, known classes only
train_pool_mask = (
    np.isin(all_groups, ["original", "extra"]) &
    np.isin(all_labels, ALL_ORIGINAL_CLASSES)
)
# Zero-day test: Conti variants
variant_mask = np.isin(all_labels, VARIANT_CLASSES)
# Mix: simultaneous executions (separate, not used in training)
mix_mask = all_groups == "mix"

print(f"\nData split:")
print(f"  Train pool (original+extra): {train_pool_mask.sum()}")
print(f"  Variant (zero-day):          {variant_mask.sum()}")
print(f"  Mix (simultaneous):          {mix_mask.sum()}")

# Encode 3-way labels for training pool
le = LabelEncoder()
pool_3way   = all_3way[train_pool_mask]
pool_labels = le.fit_transform(pool_3way)
print(f"\nLabel encoding: { {c: int(le.transform([c])[0]) for c in le.classes_} }")

# 80/20 train/val split, stratified
pool_indices = np.where(train_pool_mask)[0]
train_idx, val_idx = train_test_split(
    np.arange(len(pool_indices)), test_size=0.2,
    random_state=42, stratify=pool_labels
)
train_global = pool_indices[train_idx]
val_global   = pool_indices[val_idx]

print(f"  Train: {len(train_global)}  Val: {len(val_global)}")

train_ds = SequenceDataset(text_seqs[train_global], pool_labels[train_idx])
val_ds   = SequenceDataset(text_seqs[val_global],   pool_labels[val_idx])
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
        loss.backward()
        optimizer.step()
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
        torch.save(model.state_dict(), OUTPUT_DIR / "classifier.pt")

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6}  {avg_train:>11.4f}  {avg_val:>10.4f}  {acc:>8.2%}")

print("\nTraining complete. Best val loss: {:.4f}".format(best_val_loss))

# Load best checkpoint
model.load_state_dict(torch.load(OUTPUT_DIR / "classifier.pt", map_location=device))
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


# ── Extract penultimate embeddings for ALL executions ─────────────────────────
print("Extracting 64-d penultimate embeddings for all 1970 executions...")
all_ds     = DataLoader(
    SequenceDataset(text_seqs, np.zeros(len(text_seqs), dtype=np.int64)),
    batch_size=64, shuffle=False, num_workers=2
)
penultimate = []
with torch.no_grad():
    for batch_x, _ in all_ds:
        _, _, pooled = model(batch_x.to(device))
        penultimate.append(pooled.cpu().numpy())
penultimate = np.concatenate(penultimate, axis=0)  # (1970, 64)
print(f"Penultimate shape: {penultimate.shape}")
np.save(OUTPUT_DIR / "cnn_penultimate.npy", penultimate)


# ── Zero-day variant evaluation ───────────────────────────────────────────────
print("\nZero-Day Detection — Conti Variants")
print("=" * 65)
print(f"{'Variant':<12}  {'Pred Class':<18}  {'Max Prob':>9}  {'N':>5}")
print("-" * 55)

ransomware_idx = list(le.classes_).index("ransomware")
results_lines  = [report, "\nZero-Day Detection — Conti Variants\n" + "="*55]

for variant in VARIANT_CLASSES:
    mask = all_labels == variant
    if mask.sum() == 0:
        continue
    var_seqs = torch.tensor(text_seqs[mask], dtype=torch.float32)
    with torch.no_grad():
        logits, _, _ = model(var_seqs.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds    = probs.argmax(axis=1)
    pred_cls = le.classes_[int(np.round(np.mean(preds)))]
    avg_prob = probs.max(axis=1).mean()
    n        = mask.sum()
    line = f"{variant:<12}  {pred_cls:<18}  {avg_prob:>9.4f}  {n:>5}"
    print(line)
    results_lines.append(line)

# Save results
with open(OUTPUT_DIR / "classifier_results.txt", "w") as f:
    f.write("\n".join(results_lines))

# Save split info for downstream scripts
np.savez(OUTPUT_DIR / "fewshot_split_cnn.npz",
         train_idx=train_global,
         val_idx=val_global,
         variant_mask=variant_mask,
         mix_mask=mix_mask,
         all_labels=all_labels,
         all_groups=all_groups,
         le_classes=le.classes_)

print(f"\nSaved:")
print(f"  {OUTPUT_DIR}/classifier.pt")
print(f"  {OUTPUT_DIR}/cnn_penultimate.npy  {penultimate.shape}")
print(f"  {OUTPUT_DIR}/fewshot_split_cnn.npz")
print(f"  {OUTPUT_DIR}/classifier_results.txt")
print("\nDone.")
