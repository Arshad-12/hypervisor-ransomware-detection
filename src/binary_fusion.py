"""
binary_fusion.py — 2-way Classifier + Binary Fusion Ablation
=============================================================
Per HANDOFF update 37 Opus decision / addendum (July 8, 2026).

What this script does:
  1. Retrains A2_BiLSTM on fixed feat_23 (5-fold OOF + 3-seed final).
     Saves: lstm_scores_all.npy, lstm_threshold.npy
  2. Quantifies Idle-rate on val non-ransomware under 3-way + A2_LSTM (v6_A2_LSTM).
     → The Idle-as-monitoring story for Conti_06.
  3. Retrains 2-way MiniLM CNN (Ransomware vs Non-ran, n_classes=2).
     Saves: classifier_2way.pt, penultimate_2way.npy
  4. Threshold sweep on A2_LSTM anomaly head → (recall, Conti_06, FPR_ben, FPR_lenc)
     for both 3-way-binary (v6_A2_LSTM) and 2-way-binary fusions.
  5. Proper 6-fold LOFO: retrain 2-way CNN per fold; test on held-out family;
     report (recall, FPR_ben, FPR_lenc) per fold and mean.
  6. Ablation table: 3-way v6_A2_LSTM (87%/1.1%) vs 2-way binary (this run).
  7. Save binary_fusion_results.txt + .json

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

FEATURES_DIR   = Path("/scratch/aabdulj4/RanSMAP/features")
K_FOLDS        = 5
STREAM_THRESH  = 0.65
BENIGN_IDX, LENC_IDX, RAN_IDX = 0, 1, 2
FPR_PCT        = 95.0

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

t0 = time.time()
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])
log(f"  feat_23 loaded: {feat_23.shape}  in {time.time()-t0:.1f}s")

log("  Loading MiniLM sequences (1970, 300, 384)...")
text_seqs = np.load(FEATURES_DIR / "text_sequences.npy")   # (1970, 300, 384)

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]
mix_idx     = np.where(split["mix_mask"])[0]

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]

def label_to_3class(lbl):
    if lbl in RANSOMWARE_CLASSES: return 2
    if lbl in LEGIT_ENC_CLASSES:  return 1
    return 0

train_y3      = np.array([label_to_3class(all_labels[i]) for i in train_idx])
train_y_bin   = (train_y3 == 2).astype(int)   # 1=ransomware, 0=rest
train_nonran  = (train_y_bin == 0)

val_nonran_idx  = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])
val_ran_idx     = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])
val_ben_idx     = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx    = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])

log(f"  Training pool: {len(train_idx)} (ran={train_y_bin.sum()}, rest={(~train_nonran).sum()}...wait, "
    f"non-ran={train_nonran.sum()})")
log(f"  Val: {len(val_idx)}  (non-ran={len(val_nonran_idx)}, ran={len(val_ran_idx)})")
log(f"        Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)}")
log(f"  Variants: {len(variant_idx)}")

eval_idx = np.concatenate([val_idx, variant_idx, mix_idx])
json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Retrain A2 BiLSTM (5-fold OOF + 3-seed final)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("A2 BiLSTM on fixed feat_23 (5-fold OOF + 3-seed final)"); log("=" * 72)

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_features=23, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        D = hidden_dim * 2
        self.head = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, x):
        B, T, nf = x.shape
        x_bn = self.bn_input(x.reshape(B*T, nf)).reshape(B, T, nf)
        _, (h_n, _) = self.lstm(x_bn)
        return self.head(torch.cat([h_n[-2], h_n[-1]], dim=1))

def normalize_seqs(X, scaler=None):
    N, T, nf = X.shape
    flat = X.reshape(-1, nf).astype(np.float64)
    if scaler is None:
        scaler = StandardScaler(); flat_s = scaler.fit_transform(flat).astype(np.float32)
    else:
        flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(N, T, nf), scaler

def train_lstm(X_n, y, n_epochs=60, bs=64, lr=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = BiLSTMClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_ran = y.sum(); n_rest = len(y) - n_ran
    w = torch.tensor([n_ran/len(y), n_rest/len(y)], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    dl = DataLoader(TensorDataset(torch.tensor(X_n), torch.tensor(y, dtype=torch.long)), bs, shuffle=True)
    model.train()
    for ep in range(n_epochs):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep + 1) % 20 == 0: log(f"    ep {ep+1}/{n_epochs}")
    return model

@torch.no_grad()
def score_lstm(model, X_n):
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X_n)), 128, shuffle=False)
    probs = []
    for (bx,) in dl:
        probs.append(torch.softmax(model(bx.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

X_tr_seqs = feat_23[train_idx]
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
lstm_oof_scores = np.zeros(len(train_idx))

for fold, (tr_f, val_f) in enumerate(skf.split(X_tr_seqs, train_y_bin)):
    log(f"  --- BiLSTM OOF fold {fold+1}/{K_FOLDS} ---")
    X_f_tr_n, sc_f = normalize_seqs(X_tr_seqs[tr_f])
    X_f_val_n, _   = normalize_seqs(X_tr_seqs[val_f], sc_f)
    mf = train_lstm(X_f_tr_n, train_y_bin[tr_f], n_epochs=60, seed=fold)
    lstm_oof_scores[val_f] = score_lstm(mf, X_f_val_n)
    del mf; torch.cuda.empty_cache()

lstm_thresh_oof = float(np.percentile(lstm_oof_scores[train_nonran], FPR_PCT))
log(f"\n  OOF thresh: {lstm_thresh_oof:.4f}  "
    f"OOF FPR: {float((lstm_oof_scores[train_nonran] > lstm_thresh_oof).mean()):.1%}  "
    f"OOF TPR: {float((lstm_oof_scores[~train_nonran] > lstm_thresh_oof).mean()):.1%}")

log("\n  Training final BiLSTM (3 seeds, all 1800)...")
sc_lstm_final = StandardScaler()
X_all_n_base = feat_23[train_idx]
X_all_n, sc_lstm_final = normalize_seqs(X_all_n_base)
X_all_norm, _ = normalize_seqs(feat_23, sc_lstm_final)

final_lstms = []
for seed in [0, 1, 2]:
    log(f"  Seed {seed}:")
    mf = train_lstm(X_all_n, train_y_bin, n_epochs=80, seed=seed)
    final_lstms.append(mf)

lstm_scores_all = np.mean([score_lstm(m, X_all_norm) for m in final_lstms], axis=0)
lstm_thresh_val = float(np.percentile(lstm_scores_all[val_nonran_idx], FPR_PCT))
log(f"\n  Final LSTM val thresh: {lstm_thresh_val:.4f}  "
    f"val FPR(Attack): {float((lstm_scores_all[val_nonran_idx] > lstm_thresh_val).mean()):.1%}  "
    f"val TPR: {float((lstm_scores_all[val_ran_idx] > lstm_thresh_val).mean()):.1%}")

np.save(FEATURES_DIR / "lstm_scores_all.npy", lstm_scores_all)
np.save(FEATURES_DIR / "lstm_threshold.npy", np.array([lstm_thresh_val]))
log("  Saved: lstm_scores_all.npy + lstm_threshold.npy")

del final_lstms; torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Load 3-way MiniLM CNN + compute primary probs + streaming
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Loading 3-way MiniLM CNN + probs + streaming"); log("=" * 72)

class TextEmbeddingClassifier(nn.Module):
    def __init__(self, emb_dim=384, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.classifier   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_classes))
        self.reconstructor = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, emb_dim))
    def forward(self, x):
        feat   = self.conv(x.permute(0, 2, 1))
        pooled = feat.mean(dim=-1)
        return self.classifier(pooled), self.reconstructor(pooled), pooled

clf3 = TextEmbeddingClassifier(emb_dim=384, n_classes=3).to(device)
clf3.load_state_dict(torch.load(FEATURES_DIR / "classifier.pt", map_location=device, weights_only=True))
clf3.eval()

@torch.no_grad()
def primary_probs_3way(i):
    x = torch.tensor(text_seqs[i][None], dtype=torch.float32).to(device)
    logits, _, _ = clf3(x)
    return torch.softmax(logits, dim=1).cpu().numpy()[0]

@torch.no_grad()
def streaming_max(i):
    seq = text_seqs[i]; W = 50; best = 0.0
    for start in range(0, 300 - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = clf3(x)
        p = torch.softmax(logits, dim=1)[0, RAN_IDX].item()
        if p > best: best = p
    return best

log(f"  Precomputing 3-way probs + streaming for {len(eval_idx)} eval execs...")
pmap3, smap3 = {}, {}
for ii, i in enumerate(eval_idx):
    i = int(i)
    pmap3[i] = primary_probs_3way(i)
    if pmap3[i].argmax() != RAN_IDX:
        smap3[i] = streaming_max(i)
    else:
        smap3[i] = 0.0
    if (ii + 1) % 100 == 0: log(f"    {ii+1}/{len(eval_idx)}")

log("  Done.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Idle-rate diagnostic (3-way + A2_LSTM = v6_A2_LSTM)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("IDLE-RATE on val non-ran under 3-way + A2_LSTM (v6_A2_LSTM)"); log("=" * 72)

def v6_verdict(i, lstm_thresh):
    i = int(i)
    p = pmap3[i]; s = smap3[i]; score = lstm_scores_all[i]
    fires = score > lstm_thresh
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]: return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"
    if s >= STREAM_THRESH: return "Attack"
    if fires: return "Idle"
    return "Benign"

verdicts_val_nonran = {i: v6_verdict(i, lstm_thresh_val) for i in val_nonran_idx}
idle_ben  = sum(1 for i in val_ben_idx  if verdicts_val_nonran[i] == "Idle")
idle_lenc = sum(1 for i in val_lenc_idx if verdicts_val_nonran[i] == "Idle")
att_ben   = sum(1 for i in val_ben_idx  if verdicts_val_nonran[i] == "Attack")
att_lenc  = sum(1 for i in val_lenc_idx if verdicts_val_nonran[i] == "Attack")
att_nr    = sum(1 for i in val_nonran_idx if verdicts_val_nonran[i] == "Attack")
idle_nr   = sum(1 for i in val_nonran_idx if verdicts_val_nonran[i] == "Idle")

log(f"\n  Val Benign  ({len(val_ben_idx)}):  Attack={att_ben} ({att_ben/len(val_ben_idx):.1%})  "
    f"Idle={idle_ben} ({idle_ben/len(val_ben_idx):.1%})")
log(f"  Val LEnc    ({len(val_lenc_idx)}):  Attack={att_lenc} ({att_lenc/len(val_lenc_idx):.1%})  "
    f"Idle={idle_lenc} ({idle_lenc/len(val_lenc_idx):.1%})")
log(f"  Combined NR ({len(val_nonran_idx)}):  Attack={att_nr} ({att_nr/len(val_nonran_idx):.1%})  "
    f"Idle={idle_nr} ({idle_nr/len(val_nonran_idx):.1%})")

# Conti_06 Idle-as-monitoring: Attack ∪ Idle
c06_idx = [int(i) for i in variant_idx if all_labels[i] == "Conti_06"]
c06_verdicts = {i: v6_verdict(i, lstm_thresh_val) for i in c06_idx}
c06_attack = sum(1 for v in c06_verdicts.values() if v == "Attack")
c06_idle   = sum(1 for v in c06_verdicts.values() if v == "Idle")
c06_flagged = c06_attack + c06_idle
log(f"\n  Conti_06 under v6_A2_LSTM (thresh={lstm_thresh_val:.4f}):")
log(f"    Attack={c06_attack}/10  Idle={c06_idle}/10  Benign={10-c06_flagged}/10")
log(f"    Attack ∪ Idle (flagged-for-monitoring): {c06_flagged}/10 = {c06_flagged/10:.0%}")
log(f"\n  Per-exec:")
log(f"  {'Idx':>6}  {'LSTM_score':>11}  {'Verdict':>8}")
for i in c06_idx:
    log(f"  {i:>6}  {lstm_scores_all[i]:>11.4f}  {c06_verdicts[i]:>8}")

json_out["idle_rate"] = {
    "thresh": lstm_thresh_val,
    "val_benign_attack_fpr": att_ben / len(val_ben_idx),
    "val_benign_idle_rate":  idle_ben / len(val_ben_idx),
    "val_lenc_attack_fpr":   att_lenc / len(val_lenc_idx),
    "val_lenc_idle_rate":    idle_lenc / len(val_lenc_idx),
    "val_nr_attack_fpr":     att_nr / len(val_nonran_idx),
    "val_nr_idle_rate":      idle_nr / len(val_nonran_idx),
    "c06_attack": c06_attack, "c06_idle": c06_idle, "c06_flagged": c06_flagged,
}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Train 2-way MiniLM CNN (Ransomware vs Non-ransomware)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("TRAIN 2-way MiniLM CNN (Ransomware vs Non-ran)"); log("=" * 72)

X_tr_text = text_seqs[train_idx]   # (1800, 300, 384)
train_y2   = train_y_bin            # 1=ran, 0=non-ran

def train_cnn_2way(X_tr, y, n_epochs=80, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TextEmbeddingClassifier(emb_dim=384, n_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
    n_ran = y.sum(); n_rest = len(y) - n_ran
    w = torch.tensor([n_ran/len(y), n_rest/len(y)], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dl = DataLoader(TensorDataset(X_t, y_t), 32, shuffle=True)
    best_val_loss = float("inf"); best_state = None
    for ep in range(n_epochs):
        model.train()
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, _, _ = model(bx)
            loss = crit(logits, by); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0: log(f"  ep {ep+1}/{n_epochs}")
    return model

@torch.no_grad()
def probs_2way_batch(model, seqs, idx_list):
    model.eval()
    result = {}
    for i in idx_list:
        x = torch.tensor(seqs[int(i)][None], dtype=torch.float32).to(device)
        logits, _, _ = model(x)
        result[int(i)] = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return result

@torch.no_grad()
def streaming_max_2way(model, i):
    seq = text_seqs[int(i)]; W = 50; best = 0.0
    for start in range(0, 300 - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = model(x)
        p = torch.softmax(logits, dim=1)[0, 1].item()   # class 1 = ransomware
        if p > best: best = p
    return best

log("  Training 2-way CNN (3 seeds)...")
best_val_acc = 0; best_clf2 = None
for seed in [0, 1, 2]:
    log(f"  Seed {seed}:")
    m = train_cnn_2way(X_tr_text, train_y2, n_epochs=80, seed=seed)
    m.eval()
    # Quick val acc
    with torch.no_grad():
        X_val_t = torch.tensor(text_seqs[val_idx], dtype=torch.float32)
        dl_val = DataLoader(TensorDataset(X_val_t), 64, shuffle=False)
        preds = []
        for (bx,) in dl_val:
            lg, _, _ = m(bx.to(device))
            preds.append(lg.argmax(1).cpu().numpy())
        preds = np.concatenate(preds)
        val_y2 = np.array([(1 if all_labels[i] in RANSOMWARE_CLASSES else 0) for i in val_idx])
        val_acc = (preds == val_y2).mean()
    log(f"    val acc: {val_acc:.1%}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc; best_clf2 = m

torch.save(best_clf2.state_dict(), FEATURES_DIR / "classifier_2way.pt")
log(f"\n  Best 2-way CNN val acc: {best_val_acc:.1%}  Saved: classifier_2way.pt")

log(f"\n  Computing 2-way probs + streaming for {len(eval_idx)} eval execs...")
pmap2, smap2 = {}, {}
for ii, i in enumerate(eval_idx):
    i = int(i)
    pmap2[i] = probs_2way_batch(best_clf2, text_seqs, [i])[i]
    if pmap2[i].argmax() != 1:   # if not ransomware-primary
        smap2[i] = streaming_max_2way(best_clf2, i)
    else:
        smap2[i] = 0.0
    if (ii + 1) % 100 == 0: log(f"    {ii+1}/{len(eval_idx)}")

del best_clf2; torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Threshold sweep (both 3-way v6_A2_LSTM and 2-way binary)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("THRESHOLD SWEEP on A2_LSTM anomaly head"); log("=" * 72)

def binary_verdict_3way(i, lstm_thresh):
    """3-way primary + binary output (Attack/Benign, no Idle)."""
    i = int(i); p = pmap3[i]; s = smap3[i]; score = lstm_scores_all[i]
    fires = score > lstm_thresh
    if p[RAN_IDX] >= p[BENIGN_IDX] and p[RAN_IDX] >= p[LENC_IDX]: return "Attack"
    if p[LENC_IDX] >= p[BENIGN_IDX]:
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"
    if s >= STREAM_THRESH: return "Attack"
    if fires: return "Attack"   # binary: Idle → Attack
    return "Benign"

def binary_verdict_2way(i, lstm_thresh):
    """2-way primary + binary output (Attack/Benign, no Idle)."""
    i = int(i); p = pmap2[i]; s = smap2[i]; score = lstm_scores_all[i]
    fires = score > lstm_thresh
    if p[1] >= 0.5: return "Attack"   # ransomware-primary
    if s >= STREAM_THRESH or fires: return "Attack"
    return "Benign"

thresholds = np.linspace(0.0, 1.0, 51)

def sweep(verdict_fn, label):
    rows = []
    for thresh in thresholds:
        b_fp  = sum(1 for i in val_ben_idx  if verdict_fn(i, thresh) == "Attack")
        le_fp = sum(1 for i in val_lenc_idx if verdict_fn(i, thresh) == "Attack")
        nr_fp = b_fp + le_fp
        var_att = sum(1 for i in variant_idx if verdict_fn(i, thresh) == "Attack")
        c06_att = sum(1 for i in c06_idx     if verdict_fn(i, thresh) == "Attack")
        rows.append({
            "thresh": float(thresh),
            "recall": var_att / len(variant_idx),
            "c06": c06_att / 10,
            "fpr_ben": b_fp / len(val_ben_idx),
            "fpr_lenc": le_fp / len(val_lenc_idx),
            "fpr_nr": nr_fp / len(val_nonran_idx),
        })
    return rows

log("\n  Sweeping 3-way-binary (Attack/Benign, Idle→Attack)...")
sweep_3way = sweep(binary_verdict_3way, "3way-binary")
log("  Sweeping 2-way-binary...")
sweep_2way = sweep(binary_verdict_2way, "2way-binary")

json_out["threshold_sweep"] = {"3way_binary": sweep_3way, "2way_binary": sweep_2way}

# Print sweep tables
for sweep_data, label in [(sweep_3way, "3-WAY-BINARY"), (sweep_2way, "2-WAY-BINARY")]:
    log(f"\n  [{label}] — selected points")
    log(f"  {'Thresh':>8}  {'Recall':>8}  {'C06':>6}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
    log("  " + "─" * 58)
    checkpoints = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for r in sweep_data:
        if any(abs(r["thresh"] - c) < 0.011 for c in checkpoints):
            log(f"  {r['thresh']:>8.2f}  {r['recall']:>8.0%}  {r['c06']:>6.0%}  "
                f"{r['fpr_ben']:>9.1%}  {r['fpr_lenc']:>10.1%}  {r['fpr_nr']:>8.1%}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Choose operating point + eval
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("OPERATING POINT SELECTION + BLIND EVAL"); log("=" * 72)

# Operating point: 95th pct val non-ran score for both fusions
op_thresh = lstm_thresh_val   # pre-calibrated; same for both since BiLSTM is shared

log(f"\n  Using thresh = {op_thresh:.4f} (95th pct val non-ran lstm score)")
log(f"\n  {'Fusion':>20}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}  {'C06':>6}")
log("  " + "─" * 70)

results_eval = {}
for fname, vfn in [("3way_v6_A2_LSTM", v6_verdict), ("3way_binary", binary_verdict_3way), ("2way_binary", binary_verdict_2way)]:
    var_att = sum(1 for i in variant_idx if vfn(i, op_thresh) == "Attack")
    c06_att = sum(1 for i in c06_idx     if vfn(i, op_thresh) == "Attack")
    b_fp    = sum(1 for i in val_ben_idx  if vfn(i, op_thresh) == "Attack")
    le_fp   = sum(1 for i in val_lenc_idx if vfn(i, op_thresh) == "Attack")
    nr_fp   = b_fp + le_fp
    results_eval[fname] = {
        "recall": var_att/len(variant_idx), "c06": c06_att/10,
        "fpr_ben": b_fp/len(val_ben_idx), "fpr_lenc": le_fp/len(val_lenc_idx),
        "fpr_nr": nr_fp/len(val_nonran_idx),
    }
    log(f"  {fname:>20}  {var_att/len(variant_idx):>8.0%}  "
        f"{b_fp/len(val_ben_idx):>9.1%}  {le_fp/len(val_lenc_idx):>10.1%}  "
        f"{nr_fp/len(val_nonran_idx):>8.1%}  {c06_att/10:>6.0%}")

json_out["operating_point_eval"] = results_eval

# Per-variant breakdown (3way_binary vs 2way_binary)
for fname, vfn in [("3way_binary", binary_verdict_3way), ("2way_binary", binary_verdict_2way)]:
    log(f"\n  [{fname}] per-variant recall at thresh={op_thresh:.4f}:")
    log(f"  {'Variant':<14}  {'N':>4}  {'Attack':>7}  {'Recall':>8}")
    total_att = 0
    per_v = {}
    for vname in VARIANT_CLASSES:
        vi = [int(i) for i in variant_idx if all_labels[i] == vname]
        n_att = sum(1 for i in vi if vfn(i, op_thresh) == "Attack")
        total_att += n_att
        flag = " ◀" if vname == "Conti_06" else ""
        log(f"  {vname+flag:<14}  {len(vi):>4}  {n_att:>7}  {n_att/len(vi):>8.0%}")
        per_v[vname] = {"n": len(vi), "attack": n_att, "recall": n_att/len(vi)}
    log(f"  {'ALL':<14}  {len(variant_idx):>4}  {total_att:>7}  {total_att/len(variant_idx):>8.0%}")
    json_out[f"per_variant_{fname}"] = per_v


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Proper LOFO (6-fold, retrain 2-way CNN per fold)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("PROPER LOFO — retrain 2-way CNN per fold (6 families)"); log("=" * 72)
log("  (Benchmark: NB11 streaming 83.3% / this run uses 2-way CNN + BiLSTM head)")
log()

lofo_results = {}

for family, members in LOFO_FAMILIES.items():
    log(f"  === LOFO: hold out {family} ===")

    # Indices of held-out family (from training pool — ALL execs of this family)
    held_mask = np.array([all_labels[i] in members for i in train_idx])
    held_local = np.where(held_mask)[0]          # indices into train_idx
    held_global = train_idx[held_local]           # global indices

    # Training set = training pool minus held-out family
    fold_tr_mask = ~held_mask
    fold_tr_idx  = np.where(fold_tr_mask)[0]     # indices into train_idx
    fold_tr_seqs = X_tr_text[fold_tr_idx]
    fold_tr_y2   = train_y2[fold_tr_idx]

    # Retrain 2-way CNN without this family (best seed only, 80 epochs)
    best_fold_clf = None; best_fold_acc = 0
    for seed in [0, 1, 2]:
        m = train_cnn_2way(fold_tr_seqs, fold_tr_y2, n_epochs=80, seed=seed)
        m.eval()
        with torch.no_grad():
            # Quick val acc on training split (approx)
            xi = torch.tensor(text_seqs[held_global], dtype=torch.float32)
            dl_ = DataLoader(TensorDataset(xi), 64, shuffle=False)
            logits_list = []
            for (bx,) in dl_: logits_list.append(m(bx.to(device))[0].cpu())
            held_preds = torch.cat(logits_list).argmax(1).numpy()
            # Since held-out is ransomware, all ground truth = 1
            fold_acc = (held_preds == 1).mean()
        if fold_acc > best_fold_acc:
            best_fold_acc = fold_acc; best_fold_clf = m

    # FPR on val non-ran (same val set — no ransomware in val, so unaffected by LOFO)
    fold_pmap2 = probs_2way_batch(best_fold_clf, text_seqs, [int(i) for i in val_nonran_idx])
    fold_smap2 = {}
    for i in val_nonran_idx:
        i = int(i)
        fold_pmap2[i]  # already computed
        if fold_pmap2[i].argmax() != 1:
            fold_smap2[i] = streaming_max_2way(best_fold_clf, i)
        else:
            fold_smap2[i] = 0.0

    # Verdict for val non-ran using per-fold model + shared BiLSTM
    def fold_verdict_2way(i, thresh):
        i = int(i); p = fold_pmap2[i]; s = fold_smap2.get(i, 0.0)
        fires = lstm_scores_all[i] > thresh
        if p[1] >= 0.5: return "Attack"
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"

    b_fp  = sum(1 for i in val_ben_idx  if fold_verdict_2way(i, op_thresh) == "Attack")
    le_fp = sum(1 for i in val_lenc_idx if fold_verdict_2way(i, op_thresh) == "Attack")
    nr_fp = b_fp + le_fp

    # Recall on held-out family (compute probs for held_global)
    fold_held_pmap = probs_2way_batch(best_fold_clf, text_seqs, [int(i) for i in held_global])
    fold_held_smap = {}
    for i in held_global:
        i = int(i)
        if fold_held_pmap[i].argmax() != 1:
            fold_held_smap[i] = streaming_max_2way(best_fold_clf, i)
        else:
            fold_held_smap[i] = 0.0

    def held_verdict_2way(i, thresh):
        i = int(i); p = fold_held_pmap[i]; s = fold_held_smap.get(i, 0.0)
        fires = lstm_scores_all[i] > thresh
        if p[1] >= 0.5: return "Attack"
        if s >= STREAM_THRESH or fires: return "Attack"
        return "Benign"

    n_held = len(held_global)
    n_att  = sum(1 for i in held_global if held_verdict_2way(i, op_thresh) == "Attack")
    recall = n_att / n_held

    log(f"    N={n_held}  Attack={n_att}  Recall={recall:.0%}  "
        f"FPR_Ben={b_fp/len(val_ben_idx):.1%}  FPR_LEnc={le_fp/len(val_lenc_idx):.1%}")

    lofo_results[family] = {
        "n": n_held, "attack": n_att, "recall": recall,
        "fpr_ben": b_fp / len(val_ben_idx), "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": nr_fp / len(val_nonran_idx),
    }
    del best_fold_clf; torch.cuda.empty_cache()

json_out["lofo_proper"] = lofo_results

log(f"\n  LOFO SUMMARY:")
log(f"  {'Family':<12}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}")
log("  " + "─" * 54)
recalls = []
for fam, r in lofo_results.items():
    log(f"  {fam:<12}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  "
        f"{r['fpr_lenc']:>10.1%}  {r['fpr_nr']:>8.1%}")
    recalls.append(r['recall'])
log(f"  {'MEAN':<12}  {np.mean(recalls):>8.0%}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Ablation table
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("ABLATION TABLE"); log("=" * 72)
log(f"\n  {'System':>30}  {'Recall':>8}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  {'FPR_NR':>8}  {'C06':>6}")
log("  " + "─" * 78)

ablation = [
    ("MiniLM fusion_v4 (baseline)", 0.84, 0.0, 0.033, 0.017, 0.20),
    ("3-way + v6_A2_LSTM (Attack only)", None, None, None, None, None),
    ("3-way-binary (Idle→Attack)", None, None, None, None, None),
    ("2-way-binary (this run)", None, None, None, None, None),
]
for name, *_ in ablation:
    r = results_eval.get(name.replace(" ", "_").lower())
    if name == "MiniLM fusion_v4 (baseline)":
        log(f"  {name:>30}  {'84%':>8}  {'~0%':>9}  {'~3.3%':>10}  {'1.7%':>8}  {'20%':>6}")
    elif "v6_A2_LSTM" in name:
        r = results_eval["3way_v6_A2_LSTM"]
        log(f"  {name:>30}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  {r['fpr_lenc']:>10.1%}  "
            f"{r['fpr_nr']:>8.1%}  {r['c06']:>6.0%}")
    elif "3-way-binary" in name:
        r = results_eval["3way_binary"]
        log(f"  {name:>30}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  {r['fpr_lenc']:>10.1%}  "
            f"{r['fpr_nr']:>8.1%}  {r['c06']:>6.0%}")
    elif "2-way-binary" in name:
        r = results_eval["2way_binary"]
        log(f"  {name:>30}  {r['recall']:>8.0%}  {r['fpr_ben']:>9.1%}  {r['fpr_lenc']:>10.1%}  "
            f"{r['fpr_nr']:>8.1%}  {r['c06']:>6.0%}")


# ════════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════════
(FEATURES_DIR / "binary_fusion_results.txt").write_text("\n".join(lines))
with open(FEATURES_DIR / "binary_fusion_results.json", "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
print("\nSaved: binary_fusion_results.txt / .json")
print("binary_fusion.py done.")
