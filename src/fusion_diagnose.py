"""
Fusion v2 Diagnostic — why are 21 variants not detected?
Prints primary classifier prediction for each missed variant.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANT_CLASSES   = [f"Conti_{i:02d}" for i in range(1, 8)]
RANSOMWARE_CLASSES = ["WannaCry","Ryuk","REvil","LockBit","Darkside","Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt","SDelete","Zip"]
BENIGN_CLASSES     = ["Firefox","Office","Idle"]
CLASS_ORDER        = ["benign","legit_encryption","ransomware"]

class TextEmbeddingClassifier(nn.Module):
    def __init__(self, seq_len=300, emb_dim=384, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64,  kernel_size=5, padding=2),
            nn.BatchNorm1d(64),  nn.ReLU(),
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

text_seqs  = np.load(FEATURES_DIR/"text_sequences.npy")
split      = np.load(FEATURES_DIR/"fewshot_split_cnn.npz", allow_pickle=True)
all_labels = split["all_labels"]
variant_mask = split["variant_mask"]
variant_idx  = np.where(variant_mask)[0]

clf = TextEmbeddingClassifier().to(device)
clf.load_state_dict(torch.load(FEATURES_DIR/"classifier.pt", map_location=device, weights_only=True))
clf.eval()

print(f"\n{'Exec':>5}  {'Variant':<12}  {'Primary pred':<18}  {'P(ben)':>8}  {'P(lenc)':>8}  {'P(ran)':>8}")
print("-" * 70)

for idx in variant_idx:
    label = all_labels[idx]
    x = torch.tensor(text_seqs[idx], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = clf(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = CLASS_ORDER[p.argmax()]
    detected = "Attack" if pred == "ransomware" else f"→ {pred} (MISS)"
    print(f"{idx:>5}  {label:<12}  {detected:<18}  {p[0]:>8.3f}  {p[1]:>8.3f}  {p[2]:>8.3f}")

# Summary
print("\n── Confusion breakdown for missed variants ──")
missed_as = {"benign": [], "legit_encryption": []}
for idx in variant_idx:
    x = torch.tensor(text_seqs[idx], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = clf(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = CLASS_ORDER[p.argmax()]
    if pred != "ransomware":
        missed_as[pred].append(all_labels[idx])

print(f"  Missed as Benign     ({len(missed_as['benign'])}):      {missed_as['benign']}")
print(f"  Missed as Legit-Enc  ({len(missed_as['legit_encryption'])}): {missed_as['legit_encryption']}")
print()
print("KEY: Legit-Enc misses → Rule 1 blocks streaming (root cause if many)")
print("     Benign misses    → streaming runs but threshold not met")
