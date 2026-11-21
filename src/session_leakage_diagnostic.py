"""
session_leakage_diagnostic.py — Same-Session / Same-Sample Leakage, Train vs Validation
================================================================================================
Per HANDOFF item 59 (professor comment, annotated PDF, August 15, 2026).

The comment: "How did you form windows? How did you make sure that windows from the same
execution, malware sample, or collection session do not appear across training, calibration,
validation, and testing?"

Part 1 (window-level leakage) is already clean by construction — feat_23 is (N_exec, 300, 23)
and every split (train_idx/val_idx/variant_idx) is execution-level, so all 300 windows of a
recording always land together in one split. That just needs stating in the report, not testing.

Part 2 (session-level leakage) is the real gap: `classifier_training.py` builds the train/val
split as a plain `train_test_split(..., stratify=pool_labels)` — stratified only by the 3-way
class label, NOT grouped by collection session. Two executions of the same tool recorded minutes
apart (e.g. two Conti runs 8 minutes apart, sometimes crossing midnight — see the gap analysis
below) can land one in training, one in validation. Zero-day results (Conti_01-07 held out as
whole classes, LOFO withholds whole families) are unaffected. What's potentially affected is
threshold calibration: the 95th-pct-of-val-non-ransomware threshold could be optimistic if some
validation executions have near-duplicate siblings the model already saw in training.

★ NO RETRAINING. Uses the frozen `classifier_2way.pt` (forward pass only, not training) and the
  existing deployed `lstm_scores_all.npy` (BiLSTM main-run head). No held-out set is touched or
  changed — this only re-examines an existing threshold and reports what changes if session
  siblings are excluded from the calibration set.

Session definition: same class label + timestamps close enough to belong to one collection run.
Parsed from `metadata.csv`'s `exec_dir` column, e.g.
  .../dataset/ddr4-2133-16g/Conti/Conti-20230426_23-55-33  ->  label=Conti, ts=2023-04-26 23:55:33
Consecutive executions of the SAME label are chained into one session whenever the gap to the
previous execution is <= SESSION_GAP_MINUTES. This correctly keeps midnight-crossing runs (e.g.
23:47 -> 23:55 -> 00:03) in one session while still separating genuinely distinct collection runs
(gaps of many hours to days). The gap distribution is logged so the cutoff choice is auditable,
not asserted.
"""

import json, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT = 95.0
STREAM_THRESH = 0.65
SESSION_GAP_MINUTES = 120   # chain executions of the same label into one session if gap <= this

RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES     = ["Firefox", "Office", "Idle"]

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
# SECTION 1 — Load data + existing artifacts
# ════════════════════════════════════════════════════════════════════════════════
log("=" * 72); log("Loading data + existing frozen artifacts"); log("=" * 72)

meta_df    = pd.read_csv(FEATURES_DIR / "metadata.csv")
all_labels = np.load(FEATURES_DIR / "text_labels.npy")
text_seqs  = np.load(FEATURES_DIR / "text_sequences.npy")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
train_idx   = split["train_idx"]
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]

val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
val_nonran_idx = np.concatenate([val_ben_idx, val_lenc_idx])
conti06_idx    = np.array([i for i in variant_idx if all_labels[i] == "Conti_06"])

log(f"  N execs: {len(meta_df)}  train={len(train_idx)}  val_nonran={len(val_nonran_idx)} "
    f"(Ben={len(val_ben_idx)}, LEnc={len(val_lenc_idx)})  variants={len(variant_idx)}")

lstm_scores_all = np.load(FEATURES_DIR / "lstm_scores_all.npy")
log("  Loaded lstm_scores_all.npy (deployed BiLSTM main-run head) — no retrain.")

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
clf2.load_state_dict(torch.load(FEATURES_DIR / "classifier_2way.pt", map_location=device))
clf2.eval()
log("  classifier_2way.pt loaded (frozen, forward pass only — not training).")

@torch.no_grad()
def probs_2way(i):
    x = torch.tensor(text_seqs[int(i)][None], dtype=torch.float32).to(device)
    logits, _, _ = clf2(x)
    return torch.softmax(logits, dim=1).cpu().numpy()[0]

@torch.no_grad()
def streaming_max_2way(i):
    seq = text_seqs[int(i)]; W = 50; best = 0.0
    for start in range(0, 300 - W + 1, 5):
        chunk  = seq[start:start + W]
        padded = np.concatenate([chunk, np.zeros((300 - W, 384), np.float32)])
        x = torch.tensor(padded[None], dtype=torch.float32).to(device)
        logits, _, _ = clf2(x)
        p = torch.softmax(logits, dim=1)[0, 1].item()
        if p > best: best = p
    return best

json_out = {}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Parse collection sessions from exec_dir
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Parsing collection sessions from exec_dir"); log("=" * 72)

TS_RE = re.compile(r"-(\d{8})_(\d{2})-(\d{2})-(\d{2})$")

def parse_ts(exec_dir):
    base = Path(str(exec_dir)).name
    m = TS_RE.search(base)
    if not m:
        return None
    date_s, hh, mm, ss = m.groups()
    return datetime.strptime(f"{date_s}{hh}{mm}{ss}", "%Y%m%d%H%M%S")

meta_df["parsed_ts"] = meta_df["exec_dir"].apply(parse_ts)
n_unparsed = meta_df["parsed_ts"].isna().sum()
log(f"  Parsed timestamps: {len(meta_df) - n_unparsed}/{len(meta_df)}  (unparsed: {n_unparsed})")
if n_unparsed:
    log(f"  WARNING: {n_unparsed} exec_dir values didn't match the expected pattern — "
        f"those executions get their own singleton session (no siblings possible).")

# Chain into sessions per label, gap-threshold based. Log the gap distribution first.
all_gaps = []
for lbl, grp in meta_df.groupby("label"):
    ts_sorted = grp["parsed_ts"].dropna().sort_values().tolist()
    for a, b in zip(ts_sorted, ts_sorted[1:]):
        all_gaps.append((b - a).total_seconds() / 60.0)
all_gaps = np.array(all_gaps)
if len(all_gaps):
    log(f"\n  Inter-execution gap distribution within each label (n={len(all_gaps)} consecutive pairs):")
    for cutoff in [5, 15, 30, 60, 120, 360, 1440]:
        frac = (all_gaps <= cutoff).mean()
        log(f"    <= {cutoff:>5} min: {frac:.1%} of consecutive pairs")
    log(f"  Chosen SESSION_GAP_MINUTES = {SESSION_GAP_MINUTES} "
        f"({(all_gaps <= SESSION_GAP_MINUTES).mean():.1%} of consecutive same-label pairs fall inside one session at this cutoff)")

session_id = np.full(len(meta_df), -1, dtype=int)
next_sid = 0
for lbl, grp in meta_df.groupby("label"):
    sub = grp[grp["parsed_ts"].notna()].sort_values("parsed_ts")
    prev_ts = None
    cur_sid = None
    for idx, row in sub.iterrows():
        if prev_ts is None or (row["parsed_ts"] - prev_ts).total_seconds() / 60.0 > SESSION_GAP_MINUTES:
            cur_sid = next_sid; next_sid += 1
        session_id[idx] = cur_sid
        prev_ts = row["parsed_ts"]
    # unparsed rows for this label each get their own singleton session
    for idx in grp[grp["parsed_ts"].isna()].index:
        session_id[idx] = next_sid; next_sid += 1

n_sessions = next_sid
log(f"\n  Total sessions identified: {n_sessions} (from {len(meta_df)} executions)")

# session key = (label, session_id) — session_id is already globally unique per label-chain
train_sessions_by_label = {}
for i in train_idx:
    lbl = all_labels[i]
    train_sessions_by_label.setdefault(lbl, set()).add(int(session_id[i]))


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Flag validation non-ransomware executions with a training sibling
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Flagging validation non-ransomware execs with a same-session training sibling")
log("=" * 72)

sibling_flags = {}
for i in val_nonran_idx:
    i = int(i)
    lbl = all_labels[i]
    sid = int(session_id[i])
    has_sibling = sid in train_sessions_by_label.get(lbl, set())
    sibling_flags[i] = has_sibling

sibling_idx      = np.array([i for i, f in sibling_flags.items() if f])
sibling_free_idx = np.array([i for i, f in sibling_flags.items() if not f])

log(f"  Val non-ransomware: {len(val_nonran_idx)} total")
log(f"  WITH training sibling (same session):    {len(sibling_idx)} ({len(sibling_idx)/len(val_nonran_idx):.1%})")
log(f"  WITHOUT training sibling (sibling-free):  {len(sibling_free_idx)} ({len(sibling_free_idx)/len(val_nonran_idx):.1%})")

by_class = {}
for cls in BENIGN_CLASSES + LEGIT_ENC_CLASSES:
    cls_idx = np.array([i for i in val_nonran_idx if all_labels[i] == cls])
    if len(cls_idx) == 0: continue
    n_sib = sum(1 for i in cls_idx if sibling_flags[int(i)])
    by_class[cls] = {"n": len(cls_idx), "n_with_sibling": n_sib}
    log(f"    {cls:<10}  n={len(cls_idx):>3}  with_sibling={n_sib:>3} ({n_sib/len(cls_idx):.0%})")

json_out["exposure"] = {
    "n_val_nonran": len(val_nonran_idx),
    "n_with_sibling": len(sibling_idx),
    "n_sibling_free": len(sibling_free_idx),
    "by_class": by_class,
}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Score distributions: sibling vs sibling-free
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Score distributions — BiLSTM (deployed head) + CNN P(ransomware)"); log("=" * 72)

log("  Computing CNN P(ransomware) for all val non-ransomware execs (frozen classifier, no retrain)...")
pran = {int(i): float(probs_2way(int(i))[1]) for i in val_nonran_idx}

def dist_stats(idx_arr, score_map_or_array):
    if len(idx_arr) == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    if isinstance(score_map_or_array, dict):
        vals = np.array([score_map_or_array[int(i)] for i in idx_arr])
    else:
        vals = score_map_or_array[idx_arr]
    return {"n": len(vals), "mean": float(vals.mean()), "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95))}

log(f"\n  {'Group':<28} {'Signal':<16} {'N':>4} {'Mean':>9} {'P50':>9} {'P95':>9}")
log("  " + "-" * 80)
dist_json = {}
for grp_name, idx_arr in [("With training sibling", sibling_idx), ("Sibling-free", sibling_free_idx)]:
    s_lstm = dist_stats(idx_arr, lstm_scores_all)
    s_pran = dist_stats(idx_arr, pran)
    log(f"  {grp_name:<28} {'BiLSTM score':<16} {s_lstm['n']:>4} "
        f"{s_lstm['mean'] or 0:>9.4f} {s_lstm['p50'] or 0:>9.4f} {s_lstm['p95'] or 0:>9.4f}")
    log(f"  {'':<28} {'CNN P(ran)':<16} {s_pran['n']:>4} "
        f"{s_pran['mean'] or 0:>9.4f} {s_pran['p50'] or 0:>9.4f} {s_pran['p95'] or 0:>9.4f}")
    dist_json[grp_name] = {"bilstm": s_lstm, "cnn_p_ran": s_pran}
json_out["distributions"] = dist_json


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Recompute threshold using only sibling-free validation execs
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("Threshold recalibration — sibling-free validation only"); log("=" * 72)

thr_old = float(np.percentile(lstm_scores_all[val_nonran_idx], FPR_PCT))
if len(sibling_free_idx) >= 10:
    thr_new = float(np.percentile(lstm_scores_all[sibling_free_idx], FPR_PCT))
else:
    thr_new = None
    log(f"  WARNING: only {len(sibling_free_idx)} sibling-free execs — too few for a stable 95th pct, skipping recompute.")

log(f"  Old threshold (95th pct, full val non-ran, n={len(val_nonran_idx)}):        {thr_old:.4f}")
if thr_new is not None:
    log(f"  New threshold (95th pct, sibling-free val non-ran, n={len(sibling_free_idx)}): {thr_new:.4f}")
    log(f"  Shift: {thr_new - thr_old:+.4f} ({'higher' if thr_new > thr_old else 'lower' if thr_new < thr_old else 'unchanged'})")

def binary_verdict(p, s, lstm_score, thr):
    if p >= 0.5: return True
    if s >= STREAM_THRESH: return True
    return lstm_score > thr

def eval_at_threshold(thr):
    b_fp  = sum(1 for i in val_ben_idx  if binary_verdict(pran[int(i)],
                    0.0 if pran[int(i)] >= 0.5 else streaming_max_2way(int(i)), lstm_scores_all[int(i)], thr))
    le_fp = sum(1 for i in val_lenc_idx if binary_verdict(pran[int(i)],
                    0.0 if pran[int(i)] >= 0.5 else streaming_max_2way(int(i)), lstm_scores_all[int(i)], thr))
    n_variant = 0
    for i in variant_idx:
        i = int(i)
        p = probs_2way(i)[1]
        s = 0.0 if p >= 0.5 else streaming_max_2way(i)
        if binary_verdict(p, s, lstm_scores_all[i], thr): n_variant += 1
    n_c06 = 0
    for i in conti06_idx:
        i = int(i)
        p = probs_2way(i)[1]
        s = 0.0 if p >= 0.5 else streaming_max_2way(i)
        if binary_verdict(p, s, lstm_scores_all[i], thr): n_c06 += 1
    return {
        "fpr_ben": b_fp / len(val_ben_idx), "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": (b_fp + le_fp) / len(val_nonran_idx),
        "variant_recall": n_variant / len(variant_idx),
        "conti06_recall": n_c06 / len(conti06_idx) if len(conti06_idx) else None,
    }

log("\n  Evaluating full pipeline (classifier + streaming + BiLSTM) at OLD threshold...")
res_old = eval_at_threshold(thr_old)
log(f"    FPR_Ben={res_old['fpr_ben']:.1%}  FPR_LEnc={res_old['fpr_lenc']:.1%}  FPR_NR={res_old['fpr_nr']:.1%}  "
    f"Variant recall={res_old['variant_recall']:.0%}  Conti_06={res_old['conti06_recall']:.0%}")

json_out["threshold"] = {"old": thr_old, "new": thr_new, "eval_old": res_old}

if thr_new is not None:
    log("\n  Evaluating full pipeline at NEW (sibling-free) threshold...")
    res_new = eval_at_threshold(thr_new)
    log(f"    FPR_Ben={res_new['fpr_ben']:.1%}  FPR_LEnc={res_new['fpr_lenc']:.1%}  FPR_NR={res_new['fpr_nr']:.1%}  "
        f"Variant recall={res_new['variant_recall']:.0%}  Conti_06={res_new['conti06_recall']:.0%}")
    json_out["threshold"]["eval_new"] = res_new

    log(f"\n  Delta: FPR_NR {res_old['fpr_nr']:.1%} -> {res_new['fpr_nr']:.1%}   "
        f"Variant {res_old['variant_recall']:.0%} -> {res_new['variant_recall']:.0%}   "
        f"Conti_06 {res_old['conti06_recall']:.0%} -> {res_new['conti06_recall']:.0%}")

log()
log("=" * 72)
log("READ: if the score distributions are close between the two groups and the")
log("sibling-free threshold barely moves, session leakage is immaterial to the")
log("reported ~5% FPR / recall numbers. If sibling executions score materially")
log("lower and the sibling-free threshold is higher, the reported FPR was")
log("optimistic and this is a quantified limitation for the report.")
log("=" * 72)

out_txt  = FEATURES_DIR / "session_leakage_diagnostic_results.txt"
out_json = FEATURES_DIR / "session_leakage_diagnostic_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("session_leakage_diagnostic.py done.")
