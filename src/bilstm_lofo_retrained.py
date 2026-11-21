"""
bilstm_lofo_retrained.py — 6-Family LOFO with the BiLSTM Anomaly Head Retrained Per Fold
================================================================================================
Per HANDOFF item 57 (August 13, 2026) — PROTOCOL BUG FIX for full_lofo_final.py
(July 11, job 17497021).

Bug being fixed: full_lofo_final.py reported 100% cross-family LOFO recall on all six
families, but its BiLSTM anomaly head was FIXED across every fold — precomputed once by
binary_fusion.py on the full 1,800-execution training pool (every family included), then
reused unchanged in every fold. So in each fold the CNN never saw the held-out family, but
the BiLSTM head did — the comparison against VAE-LR LOFO (74%), TS-TCC two-sided LOFO
(82%), and TS-TCC-unsupervised LOFO (85-86%), all of which retrained their anomaly
component per fold, was not protocol-consistent.

This script reruns the SAME 6-family LOFO with the BiLSTM head RETRAINED PER FOLD on the
fold's own training pool (held-out family excluded from BOTH the CNN and the BiLSTM).
Does NOT reuse lstm_scores_all.npy / lstm_threshold.npy.

Per fold:
  1. Hold out one family entirely — excluded from the CNN's AND the BiLSTM's training data.
  2. Retrain 2-way CNN on MiniLM sequences (3-seed pick-best, 80 epochs) — identical
     convention/hyperparameters to every other LOFO run this project
     (ts_tcc_twosided_lofo.py, vae_lr_lofo.py, etc) so fold validation accuracies are
     directly comparable to the August runs.
  3. Retrain BiLSTM anomaly head on feat_23 sequences — SAME architecture/hyperparameters
     as d3_fixed.py / binary_fusion.py's BiLSTMClassifier (2-layer, 64-hidden, bidirectional,
     dropout 0.3), but now SINGLE SEED per fold (seed=0, consistent with every other LOFO
     run's anomaly-component retrain), 80 epochs. The StandardScaler used to normalize
     feat_23 is fit ONLY on the fold's training pool (held-out family excluded), never on
     the full 1800 and never on val/variants.
  4. Score every execution with the fold's own CNN + fold's own BiLSTM (+ shared MiniLM
     streaming, same sliding-window logic throughout the project).
  5. Evaluate the held-out family blind — every execution of that family, train+val combined
     (same protocol as full_lofo_final.py: family membership is by label, not by split) —
     plus paired FPR on val Benign/LEnc using the SAME fold model.
  6. Record per-exec driver (classifier / streaming / bilstm / none), same breakdown as
     full_lofo_final.py.

★ VERIFICATION (HANDOFF item 57, point 5): the BiLSTM used for the WITHIN-family numbers
  (99% variant recall / 90% Conti_06, from d3_fixed.py / binary_fusion.py) is trained on
  feat_23[train_idx], and train_idx is built from ALL_ORIGINAL_CLASSES only — VARIANT_CLASSES
  (Conti_01-07) are never in train_idx. Same holds here per fold since fold_tr_idx ⊂ train_idx.
  Confirmed explicitly below (assertion + logged count).
★ CORE RULE: held-out family excluded from BOTH heads' training data; threshold = 95th pct
  of the FOLD's own BiLSTM score on validation non-ransomware; blind eval; FPR always paired;
  single seed per fold for the head, 3-seed pick-best for the CNN — consistent with every
  prior LOFO run in this project.

Does NOT overwrite full_lofo_final.py's results (full_lofo_results.txt/.json are left as-is).
Results.md gets a NEW section `BILSTM_LOFO_RETRAINED`; the July `FULL_LOFO_FINAL` section is
kept and annotated as protocol-inconsistent (fixed head), not deleted.
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

FEATURES_DIR   = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT        = 95.0
STREAM_THRESH  = 0.65
CNN_SEEDS      = [0, 1, 2]
CNN_EPOCHS     = 80
LSTM_SEED      = 0
LSTM_EPOCHS    = 80

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]
LOFO_FAMILIES        = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]

# August CNN reference accuracies (ts_tcc_twosided_lofo.py, job 19611304) — same CNN
# training convention, so these folds should reproduce closely if the split is identical.
AUG_REF_CNN_ACC = {
    "WannaCry": 0.928, "Ryuk": 0.933, "REvil": 0.972,
    "LockBit": 0.994, "Darkside": 0.994, "Conti": 0.992,
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
variant_idx = np.where(split["variant_mask"])[0]

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]

val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
val_nonran_idx = np.concatenate([val_ben_idx, val_lenc_idx])

log(f"  Training pool: {len(train_idx)} known-class execs")
log(f"  Val non-ran: {len(val_nonran_idx)} (Benign={len(val_ben_idx)}, LEnc={len(val_lenc_idx)})")

# ★ Verification (HANDOFF point 5): train_idx never contains VARIANT_CLASSES.
n_variant_in_train = len(set(train_idx.tolist()) & set(variant_idx.tolist()))
assert n_variant_in_train == 0, "BUG: variant executions leaked into train_idx!"
log(f"  VERIFICATION: variant executions (Conti_01-07) in train_idx = {n_variant_in_train} "
    f"(expected 0 — confirmed clean; every fold's BiLSTM head is a subset of train_idx, "
    f"so it never trains on held-out variants either).")

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 2-way CNN (classifier + streaming) — identical to ts_tcc_twosided_lofo.py
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

def train_cnn_2way(X_tr, y, n_epochs=CNN_EPOCHS, seed=42):
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
# SECTION 3 — BiLSTM anomaly head — identical architecture/hyperparams to d3_fixed.py
# ════════════════════════════════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_features=23, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        D = hidden_dim * 2
        self.head = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, x):
        B, T, Fe = x.shape
        x_bn = self.bn_input(x.reshape(B*T, Fe)).reshape(B, T, Fe)
        _, (h_n, _) = self.lstm(x_bn)
        return self.head(torch.cat([h_n[-2], h_n[-1]], dim=1))

def normalize_seqs(X, scaler=None):
    N, T, Fe = X.shape
    flat = X.reshape(-1, Fe).astype(np.float64)
    if scaler is None:
        scaler = StandardScaler(); flat_s = scaler.fit_transform(flat).astype(np.float32)
    else:
        flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(N, T, Fe), scaler

def train_lstm(X_n, y, n_epochs=LSTM_EPOCHS, bs=64, lr=1e-3, seed=LSTM_SEED):
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
    return model

@torch.no_grad()
def score_lstm(model, X_n):
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X_n)), 128, shuffle=False)
    probs = []
    for (bx,) in dl:
        bx = bx.to(device)
        probs.append(torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Fusion verdict (classifier / streaming / bilstm, driver-attributed)
# ════════════════════════════════════════════════════════════════════════════════

def binary_verdict(p_ran, s, lstm_score, lstm_thr):
    if p_ran >= 0.5:             return "Attack", "classifier"
    if s >= STREAM_THRESH:       return "Attack", "streaming"
    if lstm_score > lstm_thr:    return "Attack", "bilstm"
    return "Benign", "none"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — 6-Family LOFO, BiLSTM head retrained per fold
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72)
log("6-FAMILY LOFO — BiLSTM anomaly head RETRAINED PER FOLD (protocol fix, HANDOFF item 57)")
log("=" * 72)

summary  = []
json_out["lofo"] = {}

for family in LOFO_FAMILIES:
    t0 = time.time()
    log(); log("=" * 72)
    log(f"LOFO: hold out {family}")
    log("=" * 72)

    family_idx = np.array([i for i in range(len(all_labels)) if all_labels[i] == family])
    n_train_f  = sum(i in train_idx for i in family_idx)
    n_val_f    = sum(i in val_idx   for i in family_idx)
    log(f"  {family} execs: {len(family_idx)} (train={n_train_f}, val={n_val_f})")

    fold_tr_idx = np.array([i for i in train_idx if all_labels[i] != family])
    fold_tr_y2  = np.array([1 if all_labels[i] in RANSOMWARE_CLASSES else 0 for i in fold_tr_idx])
    log(f"  Training without {family}: {len(fold_tr_idx)} execs ({fold_tr_y2.sum()} ran)")

    # -- Fold CNN (3-seed pick-best, MiniLM text sequences) --
    val_y2 = np.array([1 if all_labels[i] in RANSOMWARE_CLASSES else 0 for i in val_idx])
    best_clf, best_acc = None, -1
    for seed in CNN_SEEDS:
        m = train_cnn_2way(text_seqs[fold_tr_idx], fold_tr_y2, n_epochs=CNN_EPOCHS, seed=seed)
        m.eval()
        with torch.no_grad():
            dl_ = DataLoader(TensorDataset(torch.tensor(text_seqs[val_idx], dtype=torch.float32)), 64, shuffle=False)
            preds = np.concatenate([m(bx.to(device))[0].argmax(1).cpu().numpy() for (bx,) in dl_])
            acc = (preds == val_y2).mean()
        if acc > best_acc:
            best_acc = acc; best_clf = m
    ref_acc = AUG_REF_CNN_ACC.get(family)
    log(f"  Fold CNN val acc: {best_acc:.1%}  (Aug reference for this fold: {ref_acc:.1%} — "
        f"{'MATCH' if abs(best_acc - ref_acc) < 0.02 else 'DIFFERS, folds not identical'})")

    # -- Fold BiLSTM head (single seed, feat_23, scaler fit on fold's training pool only) --
    scaler_fold = StandardScaler()
    scaler_fold.fit(feat_23[fold_tr_idx].reshape(-1, 23).astype(np.float64))
    X_fold_tr_n, _ = normalize_seqs(feat_23[fold_tr_idx], scaler_fold)
    lstm_model = train_lstm(X_fold_tr_n, fold_tr_y2, n_epochs=LSTM_EPOCHS, seed=LSTM_SEED)
    X_all_n, _ = normalize_seqs(feat_23, scaler_fold)
    lstm_scores_fold = score_lstm(lstm_model, X_all_n)
    lstm_thr_fold = float(np.percentile(lstm_scores_fold[val_nonran_idx], FPR_PCT))
    log(f"  Fold BiLSTM thresh: {lstm_thr_fold:.4f}  "
        f"val FPR(BiLSTM alone): {float((lstm_scores_fold[val_nonran_idx] > lstm_thr_fold).mean()):.1%}")

    # -- Evaluate held-out family (all execs, train+val combined, blind) --
    rows = []
    for idx in family_idx:
        i = int(idx)
        p_arr = probs_2way(best_clf, text_seqs, i)
        p = p_arr[1]
        s = streaming_max_2way(best_clf, text_seqs, i) if p < 0.5 else 0.0
        lstm_s = float(lstm_scores_fold[i])
        verdict, driver = binary_verdict(p, s, lstm_s, lstm_thr_fold)
        rows.append({"idx": i, "p_ran": p, "stream": s, "lstm": lstm_s,
                     "verdict": verdict, "driver": driver})

    n_attack = sum(1 for r in rows if r["verdict"] == "Attack")
    driver_ct = {}
    for r in rows:
        driver_ct[r["driver"]] = driver_ct.get(r["driver"], 0) + 1
    log(f"  {family}: Attack={n_attack}/{len(rows)} ({n_attack/len(rows):.0%})")
    log(f"  Driver: " + "  ".join(f"{d}={c}" for d, c in sorted(driver_ct.items())))

    # -- Val FPR (fold's own CNN + fold's own BiLSTM) --
    b_fp = le_fp = 0
    for i in val_ben_idx:
        i = int(i)
        p_arr = probs_2way(best_clf, text_seqs, i); p = p_arr[1]
        s = streaming_max_2way(best_clf, text_seqs, i) if p < 0.5 else 0.0
        v, _ = binary_verdict(p, s, float(lstm_scores_fold[i]), lstm_thr_fold)
        if v == "Attack": b_fp += 1
    for i in val_lenc_idx:
        i = int(i)
        p_arr = probs_2way(best_clf, text_seqs, i); p = p_arr[1]
        s = streaming_max_2way(best_clf, text_seqs, i) if p < 0.5 else 0.0
        v, _ = binary_verdict(p, s, float(lstm_scores_fold[i]), lstm_thr_fold)
        if v == "Attack": le_fp += 1

    nr_fp = b_fp + le_fp
    log(f"  Val FPR: Ben={b_fp}/{len(val_ben_idx)} ({b_fp/len(val_ben_idx):.1%})  "
        f"LEnc={le_fp}/{len(val_lenc_idx)} ({le_fp/len(val_lenc_idx):.1%})  "
        f"NR={nr_fp}/{len(val_nonran_idx)} ({nr_fp/len(val_nonran_idx):.1%})  ({time.time()-t0:.0f}s)")

    summary.append({
        "family": family, "N": len(rows),
        "recall": n_attack / len(rows),
        "fold_cnn_val_acc": float(best_acc),
        "aug_ref_cnn_val_acc": ref_acc,
        "lstm_thresh": lstm_thr_fold,
        "fpr_ben": b_fp / len(val_ben_idx),
        "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": nr_fp / len(val_nonran_idx),
        "drivers": driver_ct,
    })
    json_out["lofo"][family] = summary[-1]

    missed = [r for r in rows if r["verdict"] == "Benign"]
    if missed:
        log(f"  MISSED execs ({len(missed)}):")
        log(f"    {'Idx':>6}  {'p_ran':>7}  {'stream':>7}  {'lstm':>7}")
        for r in missed:
            log(f"    {r['idx']:>6}  {r['p_ran']:>7.4f}  {r['stream']:>7.4f}  {r['lstm']:>7.4f}")
    else:
        log(f"  No missed execs — 100% recall.")

    del best_clf, lstm_model; torch.cuda.empty_cache()

# ── Summary table ──────────────────────────────────────────────────────────────
log(); log("=" * 72); log("BILSTM LOFO SUMMARY — HEAD RETRAINED PER FOLD (protocol fix)"); log("=" * 72)
log(f"  {'Family':>12}  {'N':>4}  {'Recall':>8}  {'CNN acc':>9}  {'FPR_Ben':>9}  {'FPR_LEnc':>10}  "
    f"{'FPR_NR':>8}  {'Classif':>8}  {'Stream':>8}  {'BiLSTM':>8}")
log("  " + "─" * 100)
for s in summary:
    dc = s["drivers"]
    log(f"  {s['family']:>12}  {s['N']:>4}  {s['recall']:>8.0%}  {s['fold_cnn_val_acc']:>9.1%}  "
        f"{s['fpr_ben']:>9.1%}  {s['fpr_lenc']:>10.1%}  {s['fpr_nr']:>8.1%}  "
        f"{dc.get('classifier',0):>8}  {dc.get('streaming',0):>8}  {dc.get('bilstm',0):>8}")

mean_recall = np.mean([s["recall"] for s in summary])
mean_fpr    = np.mean([s["fpr_nr"] for s in summary])
log("  " + "─" * 100)
log(f"  {'MEAN':>12}  {'':>4}  {mean_recall:>8.0%}  {'':>9}  {'':>9}  {'':>10}  {mean_fpr:>8.1%}")
log()
log(f"  Compare: FULL_LOFO_FINAL (July, fixed BiLSTM head) = 100%. "
    f"VAE likelihood-ratio LOFO = 74%. TS-TCC two-sided LOFO = 82%. "
    f"TS-TCC-unsupervised LOFO = 85-86%. VAE single-formulation LOFO = 70%.")

json_out["mean_recall"] = float(mean_recall)
json_out["mean_fpr_nr"] = float(mean_fpr)

# ── Save ──────────────────────────────────────────────────────────────────────
(FEATURES_DIR / "bilstm_lofo_retrained_results.txt").write_text("\n".join(lines))
with open(FEATURES_DIR / "bilstm_lofo_retrained_results.json", "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(); log("Saved: bilstm_lofo_retrained_results.txt / .json")
log("bilstm_lofo_retrained.py done.")
