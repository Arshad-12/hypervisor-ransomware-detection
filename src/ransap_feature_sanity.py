"""
ransap_feature_sanity.py — Is the T_sr/T_sw Scale Mismatch Actually Hurting BiLSTM?
================================================================================================
Per HANDOFF item 62 (August 17, 2026) — PREREQUISITE for item 61.

Item 61 found T_sr/T_sw (raw I/O-operation counts) are ~900-965x larger in RanSAP than
RanSMAP — a genuine accounting difference between the two extraction pipelines, not a fixable
unit conversion. TS-TCC and the VAE (distance/reconstruction-based heads) saturated to 100% on
EVERY RanSAP category because of it. The BiLSTM did NOT saturate in the original
RANSAP_CROSS_DATASET run (82.8% recall / 5.8% standard-benign FPR) — but it was scored on the
same two broken features. A sigmoid output can absorb an extreme input dimension instead of
blowing up on it, so the 82.8% may be resting entirely on the three comparable features (V_sr,
V_sw, H_sw) with T_sr/T_sw effectively ignored — or it may be quietly unreliable. This decides
whether Section 7.7/7.8 of the report survive, and which form item 61 should take.

★ IMPORTANT — neither of the two BiLSTM heads this task needs was ever persisted to disk:
  - `ransap_eval.py`'s storage-only 5-dim BiLSTM (`RANSAP_CROSS_DATASET`, job 17697336): trained
    fresh every run, no checkpoint or scaler saved.
  - `d3_fixed.py`/`binary_fusion.py`'s deployed 23-dim BiLSTM (the "99%/90%" within-family head):
    only its SCORES (`lstm_scores_all.npy`) were saved, not the model weights or scaler.
  So this script does ONE reproduction of EACH, using the exact same architecture, seeds, and
  hyperparameters as the original scripts (verbatim copies), and validates each reproduction
  against the already-recorded reference numbers BEFORE trusting the zeroed-feature comparison.
  This is reproduction of an existing result for scoring purposes, not a new experiment or any
  retuning — no threshold is refit, no scaler is refit on RanSAP, nothing is fit on RanSAP at all.

PART A — RanSAP cross-dataset (storage-only 5-dim BiLSTM):
  1. Reproduce the exact `ransap_eval.py` BiLSTM (3-seed ensemble, SEEDS=[42,123,777], 50 epochs,
     threshold calibrated on RanSMAP TRAINING non-ransomware — same protocol as the original).
  2. Score RanSAP with the ORIGINAL unmodified 5-dim features — sanity check against the
     recorded 82.8% recall / Firefox 1.7% / Excel 9.8% / SDelete 8.3% / Zip 88.3% / AESCrypt 93.3%.
  3. Zero T_sr and T_sw (columns 0, 1) in the RanSAP feat5 ONLY (leave RanSMAP training
     untouched), re-score with the SAME model + SAME scaler + SAME threshold. Report side by side.

PART B — RanSMAP control (deployed 23-dim BiLSTM, within-family):
  1. Reproduce the exact `d3_fixed.py` BiLSTM (3-seed ensemble, SEEDS=[0,1,2], 80 epochs, full
     23-dim feat_23, threshold = 95th pct RanSMAP val non-ransomware).
  2. Score the 70 Conti variants + Conti_06 with the ORIGINAL unmodified 23-dim features —
     sanity check against the recorded 99% variant recall / 90% Conti_06.
  3. Zero T_sr and T_sw (columns 0, 1 of feat_23) for the variant/Conti_06 executions being
     scored ONLY (training untouched), re-score with the SAME model + SAME threshold. This shows
     whether the deployed head uses those two features AT ALL when everything (hardware, OS,
     dataset) is held constant — the cleanest way to separate "head ignores these features
     generally" from "head ignores them only because of the RanSAP scale mismatch."

★-clean: no retraining beyond the two required reproductions, no threshold refit, nothing
  fitted on RanSAP, RanSMAP training pool never touched by the zeroing (only the scored
  executions are modified).
"""

import json, re, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

RANSAP_FEAT  = Path("/scratch/aabdulj4/RanSAP/features")
RANSMAP_FEAT = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT       = 95.0
STREAM_THRESH = 0.65
CNN_THRESH    = 0.5
N_WINDOWS     = 300

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES

RAN_LABELS    = ["Cerber", "Darkside", "GandCrab4", "Ryuk", "Sodinokibi", "TeslaCrypt", "WannaCry"]
BENIGN_LABELS = ["AESCrypt", "Excel", "Firefox", "SDelete", "Zip"]

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

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# Shared: frozen classifier_2way.pt
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

cnn_clf = TextEmbeddingClassifier(emb_dim=384, n_classes=2).to(device)
cnn_clf.load_state_dict(torch.load(RANSMAP_FEAT / "classifier_2way.pt", map_location=device))
cnn_clf.eval()
log("classifier_2way.pt loaded (frozen, shared by both parts).")

@torch.no_grad()
def cnn_scores(text_seqs, indices, batch=64):
    idx_list = list(indices); all_scores = []
    for b in range(0, len(idx_list), batch):
        bidx = idx_list[b:b+batch]
        X = np.stack([text_seqs[int(i)] for i in bidx])
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        logits, _, _ = cnn_clf(Xt)
        all_scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(all_scores)

@torch.no_grad()
def streaming_scores(text_seqs, indices):
    W = 50; scores = []
    for i in indices:
        i = int(i); seq = text_seqs[i]; best = 0.0
        for start in range(0, N_WINDOWS - W + 1, 5):
            chunk  = seq[start:start + W]
            padded = np.concatenate([chunk, np.zeros((N_WINDOWS - W, text_seqs.shape[-1]), np.float32)])
            x = torch.tensor(padded[None], dtype=torch.float32).to(device)
            logits, _, _ = cnn_clf(x)
            p = torch.softmax(logits, dim=1)[0, 1].item()
            if p > best: best = p
        scores.append(best)
    return np.array(scores, dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════════
# PART A — RanSAP cross-dataset, storage-only 5-dim BiLSTM (exact ransap_eval.py repro)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("PART A — RanSAP cross-dataset, storage-only 5-dim BiLSTM"); log("=" * 72)

FEAT_DIM_A, SEEDS_A, EPOCHS_A, BATCH_A, LR_A, FPR_TARGET_A = 5, [42, 123, 777], 50, 32, 1e-3, 0.05

class BiLSTM5(nn.Module):
    """Exact architecture from ransap_eval.py."""
    def __init__(self, input_dim=FEAT_DIM_A, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(hidden * 2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        g = self.pool(out.permute(0, 2, 1)).squeeze(-1)
        return torch.sigmoid(self.fc(g)).squeeze(-1)

smap_meta   = pd.read_csv(RANSMAP_FEAT / "metadata.csv")
smap_labels = np.load(RANSMAP_FEAT / "text_labels.npy", allow_pickle=True).astype(str)
smap_text_seqs = np.load(RANSMAP_FEAT / "text_sequences.npy")
split      = np.load(RANSMAP_FEAT / "fewshot_split_cnn.npz", allow_pickle=True)
train_idx  = split["train_idx"]
val_idx    = split["val_idx"]

log("  Loading RanSMAP feat_23 (cols 0:5)...")
t0 = time.time()
smap_feat23 = np.zeros((len(smap_meta), N_WINDOWS, 23), dtype=np.float32)
for ii, row in smap_meta.iterrows():
    fp = row.get("filepath", row.get("feat_path", None))
    if fp and Path(fp).exists():
        smap_feat23[ii] = np.load(fp)
smap_feat5 = smap_feat23[:, :, :FEAT_DIM_A]
log(f"  Loaded in {time.time()-t0:.1f}s")

def is_smap_ran(lbl):
    return any(lbl == r or lbl.startswith(r + "_") or lbl.startswith(r + "-") for r in RANSOMWARE_CLASSES)
def is_smap_nonran_known(lbl):
    return lbl in ALL_ORIGINAL_CLASSES and not is_smap_ran(lbl)

train_ran_idx_a    = np.array([i for i in train_idx if is_smap_ran(smap_labels[i])])
train_nonran_idx_a = np.array([i for i in train_idx if is_smap_nonran_known(smap_labels[i])])
log(f"  RanSMAP train ran={len(train_ran_idx_a)}, non-ran={len(train_nonran_idx_a)}")

def train_bilstm5(X_ran, X_ben, seeds=SEEDS_A, epochs=EPOCHS_A):
    n_ran, n_ben = len(X_ran), len(X_ben)
    y_ran = np.ones(n_ran, dtype=np.float32); y_ben = np.zeros(n_ben, dtype=np.float32)
    X_all = np.concatenate([X_ran, X_ben], axis=0); y_all = np.concatenate([y_ran, y_ben])
    scaler = StandardScaler(); scaler.fit(X_all.reshape(-1, FEAT_DIM_A))
    X_scaled = ((X_all.reshape(-1, FEAT_DIM_A) - scaler.mean_) / (scaler.scale_ + 1e-8)).reshape(-1, N_WINDOWS, FEAT_DIM_A)
    models = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = BiLSTM5().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR_A, weight_decay=1e-4)
        dl = DataLoader(TensorDataset(torch.tensor(X_scaled, dtype=torch.float32),
                                       torch.tensor(y_all, dtype=torch.float32)), BATCH_A, shuffle=True)
        pos_w = (n_ben / n_ran) if n_ran > 0 else 1.0
        crit = nn.BCELoss(reduction="none")
        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                score = model(xb)
                w = torch.where(yb > 0.5, torch.tensor(pos_w, device=device), torch.tensor(1.0, device=device))
                loss = (crit(score, yb) * w).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        models.append(model)
    model_scores = []
    for model in models:
        model.eval()
        with torch.no_grad():
            Xt_b = torch.tensor(X_scaled[len(X_ran):], dtype=torch.float32).to(device)
            model_scores.append(model(Xt_b).cpu().numpy())
    ben_scores = np.stack(model_scores).mean(axis=0)
    lstm_thresh = float(np.percentile(ben_scores, 100 * (1 - FPR_TARGET_A)))
    return models, scaler, lstm_thresh

def ensemble_score5(models, X_scaled):
    scores = []
    for model in models:
        model.eval()
        with torch.no_grad():
            scores.append(model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy())
    return np.stack(scores).mean(axis=0)

log("  Reproducing storage-only BiLSTM (3-seed ensemble, exact ransap_eval.py config)...")
t0 = time.time()
models_a, scaler_a, thresh_a = train_bilstm5(smap_feat5[train_ran_idx_a], smap_feat5[train_nonran_idx_a])
log(f"  Done in {time.time()-t0:.0f}s. Threshold: {thresh_a:.4f}")

# RanSAP
rsp_feat5      = np.load(RANSAP_FEAT / "ransap_feat5.npy")
rsp_text_seqs  = np.load(RANSAP_FEAT / "ransap_text_seqs.npy")
rsp_labels_raw = np.load(RANSAP_FEAT / "ransap_labels.npy", allow_pickle=True).astype(str)
def norm_label(lbl): return re.sub(r'-(largefiles|w10dirs.*)$', '', lbl)
rsp_labels = np.array([norm_label(l) for l in rsp_labels_raw])
rsp_role = np.full(len(rsp_labels), "unknown", dtype=object)
for i, lbl in enumerate(rsp_labels):
    if lbl in RAN_LABELS: rsp_role[i] = "ransomware"
    elif lbl in BENIGN_LABELS: rsp_role[i] = "benign"
rsp_active = np.where(rsp_role != "unknown")[0]

p_cnn_rsp = cnn_scores(rsp_text_seqs, rsp_active)
need_stream = np.where(p_cnn_rsp < CNN_THRESH)[0]
s_stream_rsp = np.zeros(len(rsp_active), dtype=np.float32)
if len(need_stream):
    s_stream_rsp[need_stream] = streaming_scores(rsp_text_seqs, rsp_active[need_stream])

def binary_verdict(p, s, head_score, thr):
    if p >= CNN_THRESH: return True
    if s >= STREAM_THRESH: return True
    return head_score > thr

def score_ransap_variant(feat5_variant, tag):
    flat = feat5_variant.reshape(-1, FEAT_DIM_A)
    scaled = ((flat - scaler_a.mean_) / (scaler_a.scale_ + 1e-8)).reshape(-1, N_WINDOWS, FEAT_DIM_A)
    head_scores = ensemble_score5(models_a, scaled[rsp_active])
    log(f"\n  ── {tag} ──")
    log(f"  {'Family':<14} {'N':>5} {'Det':>5} {'Recall':>8}")
    ran_families = sorted(set(rsp_labels[rsp_active]) & set(RAN_LABELS))
    recall = {}
    for fam in ran_families:
        fam_mask = rsp_labels[rsp_active] == fam
        n = int(fam_mask.sum())
        det = sum(1 for j in np.where(fam_mask)[0]
                   if binary_verdict(p_cnn_rsp[j], s_stream_rsp[j], head_scores[j], thresh_a))
        recall[fam] = {"n": n, "det": int(det), "recall": det / n if n else None}
        log(f"  {fam:<14} {n:>5} {det:>5} {recall[fam]['recall']:>7.1%}")
    total_n = sum(r["n"] for r in recall.values()); total_det = sum(r["det"] for r in recall.values())
    log(f"  {'TOTAL':<14} {total_n:>5} {total_det:>5} {total_det/total_n:>7.1%}")
    fpr = {}
    log(f"  FPR:")
    for fam in ["Firefox", "Excel", "SDelete", "Zip", "AESCrypt"]:
        fam_mask = rsp_labels[rsp_active] == fam
        n = int(fam_mask.sum())
        if n == 0: continue
        fp = sum(1 for j in np.where(fam_mask)[0]
                  if binary_verdict(p_cnn_rsp[j], s_stream_rsp[j], head_scores[j], thresh_a))
        fpr[fam] = {"n": n, "fp": int(fp), "fpr": fp / n if n else None}
        log(f"    {fam:<12} {n:>5} {fp:>5} {fpr[fam]['fpr']:>7.1%}")
    return {"recall": recall, "total_recall": total_det/total_n, "fpr": fpr}

json_out["A_original"] = score_ransap_variant(rsp_feat5, "ORIGINAL (unmodified 5 features) — sanity check vs recorded 82.8%")

rsp_feat5_zeroed = rsp_feat5.copy()
rsp_feat5_zeroed[:, :, 0] = 0.0  # T_sr
rsp_feat5_zeroed[:, :, 1] = 0.0  # T_sw
json_out["A_zeroed"] = score_ransap_variant(rsp_feat5_zeroed, "T_sr/T_sw ZEROED (same model, same scaler, same threshold)")

del models_a; torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# PART B — RanSMAP control, deployed 23-dim BiLSTM (exact d3_fixed.py repro)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("PART B — RanSMAP control, deployed 23-dim BiLSTM (within-family)"); log("=" * 72)

SEEDS_B, EPOCHS_B = [0, 1, 2], 80

class BiLSTMClassifier(nn.Module):
    """Exact architecture from d3_fixed.py — the deployed 23-dim head."""
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

def train_lstm_b(X_n, y, n_epochs=EPOCHS_B, bs=64, lr=1e-3, seed=0):
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
def score_lstm_b(model, X_n):
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X_n)), 128, shuffle=False)
    probs = []
    for (bx,) in dl:
        bx = bx.to(device)
        probs.append(torch.softmax(model(bx), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

train_y_bin_b = np.array([1 if is_smap_ran(smap_labels[i]) else 0 for i in train_idx])
X_train_n, scaler_b = normalize_seqs(smap_feat23[train_idx])
log("  Reproducing deployed 23-dim BiLSTM (3-seed ensemble, exact d3_fixed.py config)...")
t0 = time.time()
models_b = [train_lstm_b(X_train_n, train_y_bin_b, seed=s) for s in SEEDS_B]
log(f"  Done in {time.time()-t0:.0f}s")

variant_idx = np.where(split["variant_mask"])[0]
conti06_idx = np.array([i for i in variant_idx if smap_labels[i] == "Conti_06"])
val_ben_idx  = np.array([i for i in val_idx if smap_labels[i] in BENIGN_CLASSES])
val_lenc_idx = np.array([i for i in val_idx if smap_labels[i] in LEGIT_ENC_CLASSES])
val_nonran_idx = np.concatenate([val_ben_idx, val_lenc_idx])

X_all_n_orig, _ = normalize_seqs(smap_feat23, scaler_b)
lstm_scores_orig = np.mean([score_lstm_b(m, X_all_n_orig) for m in models_b], axis=0)
thresh_b = float(np.percentile(lstm_scores_orig[val_nonran_idx], FPR_PCT))
log(f"  Threshold (95th pct val non-ran): {thresh_b:.4f}")

p_cnn_smap = {int(i): cnn_scores(smap_text_seqs, [i])[0] for i in np.concatenate([variant_idx, val_idx])}
s_stream_smap = {}
for i in np.concatenate([variant_idx, val_idx]):
    i = int(i)
    s_stream_smap[i] = 0.0 if p_cnn_smap[i] >= CNN_THRESH else streaming_scores(smap_text_seqs, [i])[0]

def eval_variants(lstm_scores, tag):
    n_variant = sum(1 for i in variant_idx if binary_verdict(p_cnn_smap[int(i)], s_stream_smap[int(i)], lstm_scores[int(i)], thresh_b))
    n_c06     = sum(1 for i in conti06_idx if binary_verdict(p_cnn_smap[int(i)], s_stream_smap[int(i)], lstm_scores[int(i)], thresh_b))
    b_fp  = sum(1 for i in val_ben_idx  if binary_verdict(p_cnn_smap[int(i)], s_stream_smap[int(i)], lstm_scores[int(i)], thresh_b))
    le_fp = sum(1 for i in val_lenc_idx if binary_verdict(p_cnn_smap[int(i)], s_stream_smap[int(i)], lstm_scores[int(i)], thresh_b))
    variant_recall = n_variant / len(variant_idx)
    c06_recall     = n_c06 / len(conti06_idx) if len(conti06_idx) else None
    fpr_nr = (b_fp + le_fp) / len(val_nonran_idx)
    log(f"\n  ── {tag} ──")
    log(f"    Variant recall={variant_recall:.0%}  Conti_06={c06_recall:.0%}  "
        f"FPR_Ben={b_fp/len(val_ben_idx):.1%}  FPR_LEnc={le_fp/len(val_lenc_idx):.1%}  FPR_NR={fpr_nr:.1%}")
    return {"variant_recall": variant_recall, "conti06_recall": c06_recall, "fpr_nr": fpr_nr}

json_out["B_original"] = eval_variants(lstm_scores_orig, "ORIGINAL (unmodified 23 features) — sanity check vs recorded 99%/90%")

smap_feat23_zeroed = smap_feat23.copy()
smap_feat23_zeroed[:, :, 0] = 0.0  # T_sr
smap_feat23_zeroed[:, :, 1] = 0.0  # T_sw
X_all_n_zeroed, _ = normalize_seqs(smap_feat23_zeroed, scaler_b)
lstm_scores_zeroed = np.mean([score_lstm_b(m, X_all_n_zeroed) for m in models_b], axis=0)
json_out["B_zeroed"] = eval_variants(lstm_scores_zeroed, "T_sr/T_sw ZEROED (same model, same scaler, same threshold)")

del models_b; torch.cuda.empty_cache()

log()
log("=" * 72); log("READ"); log("=" * 72)
log("  (a) RanSAP numbers barely move when T_sr/T_sw are zeroed -> BiLSTM was effectively")
log("      ignoring those two features; 82.8% stands as a 3-effective-feature result.")
log("  (b) RanSAP numbers move materially -> the original cross-dataset result is not")
log("      trustworthy as reported.")
log("  RanSMAP control isolates whether T_sr/T_sw matter to this head AT ALL (same hardware,")
log("  same dataset) vs only mattering because of the RanSAP scale mismatch specifically.")

out_txt  = RANSMAP_FEAT / "ransap_feature_sanity_results.txt"
out_json = RANSMAP_FEAT / "ransap_feature_sanity_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("ransap_feature_sanity.py done.")
