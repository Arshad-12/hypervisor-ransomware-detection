"""
channel_ablation_table.py — Complete Channel-Ablation Table (professor comment)
================================================================================================
Per HANDOFF item 65 (August 2026).

The comment: "Add a table to collect the results of channel-ablation like classifier only,
streaming only, VAE only, classifier and VAE, and full fusion. Report recall, FPR, and
Conti_06 recall."

`COMPONENT_ABLATION_FINAL` (item 58 Task A) already has classifier-only, classifier+streaming,
and the three "classifier + head" (full fusion) rows. Two rows were missing:
  - "X only" (head alone, no classifier gate, no streaming) for VAE, TS-TCC, and BiLSTM. Never
    computed before — TS-TCC has a standalone number from TS_TCC_DETECTION but the VAE and
    BiLSTM never did, so the comparison was asymmetric.
  - "streaming only" — NOT computed here. Item 63 already proved every one of 430 evaluated
    executions scores exactly 0.0000 on the streaming channel, so a streaming-only detector
    fires on nothing: 0% recall / 0% Conti_06 / 0% FPR, by construction. Derived, not measured.

★ NO RETRAINING, NO GPU NEEDED. Pure post-hoc arithmetic over three already-saved per-exec
  score arrays (vae_nonran_window_scores.npy, ts_tcc_unsup_window_scores.npy,
  lstm_scores_all.npy) plus the already-recorded classifier-only / classifier+head numbers
  from item 58. Threshold recomputed fresh here (95th pct val non-ransomware) as a consistency
  check against the values already used in COMPONENT_ABLATION_FINAL and HELDOUT_FPR_SPLIT.
"""

import json
import numpy as np
from pathlib import Path

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
FPR_PCT = 95.0

RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES     = ["Firefox", "Office", "Idle"]

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        return super().default(o)

log("=" * 72); log("Loading data + existing score arrays (no retrain, no GPU)"); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]

val_ben_idx    = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx   = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
val_nonran_idx = np.concatenate([val_ben_idx, val_lenc_idx])
conti06_idx    = np.array([i for i in variant_idx if all_labels[i] == "Conti_06"])
log(f"  Val Benign={len(val_ben_idx)}  Val LEnc={len(val_lenc_idx)}  "
    f"Variants={len(variant_idx)} (Conti_06={len(conti06_idx)})")

vae_scores   = np.load(FEATURES_DIR / "vae_nonran_window_scores.npy")
tstcc_scores = np.load(FEATURES_DIR / "ts_tcc_unsup_window_scores.npy")
lstm_scores  = np.load(FEATURES_DIR / "lstm_scores_all.npy")
log("  Loaded vae_nonran_window_scores.npy, ts_tcc_unsup_window_scores.npy, lstm_scores_all.npy")

json_out = {}


def eval_standalone(name, scores):
    thr = float(np.percentile(scores[val_nonran_idx], FPR_PCT))
    n_variant = int((scores[variant_idx] > thr).sum())
    n_c06     = int((scores[conti06_idx] > thr).sum())
    b_fp      = int((scores[val_ben_idx] > thr).sum())
    le_fp     = int((scores[val_lenc_idx] > thr).sum())
    row = {
        "recall": n_variant / len(variant_idx),
        "c06": n_c06 / len(conti06_idx) if len(conti06_idx) else None,
        "fpr_ben": b_fp / len(val_ben_idx), "fpr_lenc": le_fp / len(val_lenc_idx),
        "fpr_nr": (b_fp + le_fp) / len(val_nonran_idx), "thr": thr,
    }
    log(f"  {name:<20} thr={thr:>9.4f}  recall={row['recall']:.0%}  C06={row['c06']:.0%}  "
        f"FPR_Ben={row['fpr_ben']:.1%}  FPR_LEnc={row['fpr_lenc']:.1%}  FPR_NR={row['fpr_nr']:.1%}")
    return row

log(); log("=" * 72); log("Standalone head rows (score > threshold, NO classifier gate)"); log("=" * 72)
json_out["vae_only"]    = eval_standalone("VAE only",    vae_scores)
json_out["tstcc_only"]  = eval_standalone("TS-TCC only", tstcc_scores)
json_out["bilstm_only"] = eval_standalone("BiLSTM only", lstm_scores)

# Derived, not measured — item 63 proved streaming is exactly 0.0 on every execution.
json_out["streaming_only"] = {"recall": 0.0, "c06": 0.0, "fpr_ben": 0.0, "fpr_lenc": 0.0, "fpr_nr": 0.0}

# Already recorded (item 58, COMPONENT_ABLATION_FINAL) — not recomputed, just consolidated here.
json_out["classifier_only"] = {"recall": 0.86, "c06": 0.10, "fpr_ben": 0.0, "fpr_lenc": 0.0, "fpr_nr": 0.0}
json_out["classifier_plus_vae"]    = {"recall": 0.89, "c06": 0.30, "fpr_ben": 0.022, "fpr_lenc": 0.078, "fpr_nr": 0.050}
json_out["classifier_plus_tstcc"]  = {"recall": 0.94, "c06": 0.60, "fpr_ben": 0.033, "fpr_lenc": 0.067, "fpr_nr": 0.050}
json_out["classifier_plus_bilstm"] = {"recall": 0.99, "c06": 0.90, "fpr_ben": 0.078, "fpr_lenc": 0.022, "fpr_nr": 0.050}

log(); log("=" * 72); log("CONSOLIDATED CHANNEL-ABLATION TABLE"); log("=" * 72)
row_order = [
    ("classifier_only", "Classifier only"),
    ("streaming_only", "Streaming only (derived)"),
    ("vae_only", "VAE only"),
    ("tstcc_only", "TS-TCC only"),
    ("bilstm_only", "BiLSTM only"),
    ("classifier_plus_vae", "Classifier + VAE"),
    ("classifier_plus_tstcc", "Classifier + TS-TCC"),
    ("classifier_plus_bilstm", "Classifier + BiLSTM (full fusion)"),
]
log(f"  {'Row':<34} {'Recall':>7} {'Conti06':>8} {'FPR_Ben':>8} {'FPR_LEnc':>9} {'FPR_NR':>7}")
log("  " + "-" * 78)
for key, label in row_order:
    r = json_out[key]
    log(f"  {label:<34} {r['recall']:>7.0%} {r['c06']:>8.0%} {r['fpr_ben']:>8.1%} "
        f"{r['fpr_lenc']:>9.1%} {r['fpr_nr']:>7.1%}")

log()
log("NOTE: once streaming is confirmed removed from the deployed pipeline, 'Classifier + X'")
log("rows above ARE the full-fusion numbers already reported elsewhere (streaming is inert —")
log("relabel rather than re-run).")

out_txt  = FEATURES_DIR / "channel_ablation_table_results.txt"
out_json = FEATURES_DIR / "channel_ablation_table_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("channel_ablation_table.py done.")
