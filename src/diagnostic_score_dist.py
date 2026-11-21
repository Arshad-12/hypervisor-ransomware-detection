"""
diagnostic_score_dist.py — Per-Class VAE Score Distribution: All-Classes vs Non-Ran-Only
================================================================================================
Per HANDOFF item 55, TASK A (August 10, 2026).

Pure post-hoc analysis — NO training, NO GPU needed. Loads the already-saved
per-execution MSE score arrays from the two existing VAE runs:
  vae_fixed_window_scores.npy   — all-classes VAE (D3-FIXED / FLAT_VAE_BINARY_FUSION)
  vae_nonran_window_scores.npy  — non-ransomware-only VAE (VAE_NONRAN_ONLY)
and reports the per-class score distribution (N, mean, P50, P95, >thresh) for
Val Benign / Val LEnc / Val Ransomware / Conti_06 for BOTH models side by
side, plus the Conti_06 per-execution MSE table for the non-ran-only model
against the already-recorded all-classes values.

Tests the training-set-independence hypothesis from item 55: if Val
Ransomware MSE and the per-exec Conti_06 values are near-identical between
the two models, reconstruction error in this feature space does not depend
much on what the VAE was trained on.
"""

import json
import numpy as np
from pathlib import Path

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")

RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES     = ["Firefox", "Office", "Idle"]

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

# Reference: all-classes VAE per-exec Conti_06 MSE, already recorded in Results.md
# (FLAT_VAE_BINARY_FUSION, job 17962744).
REF_ALLCLASS_C06 = {
    1490: 1.7689, 1491: 0.9997, 1492: 6.9514, 1493: 0.8042, 1494: 0.6510,
    1495: 0.7090, 1496: 0.8278, 1497: 1.0017, 1498: 0.6110, 1499: 0.8974,
}

log("=" * 72); log("Loading data + score arrays"); log("=" * 72)

all_labels = np.load(FEATURES_DIR / "text_labels.npy")
split      = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]

val_ben_idx  = np.array([i for i in val_idx if all_labels[i] in BENIGN_CLASSES])
val_lenc_idx = np.array([i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES])
val_ran_idx  = np.array([i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES])
c06_idx      = [int(i) for i in variant_idx if all_labels[i] == "Conti_06"]

log(f"  Val Benign={len(val_ben_idx)}  Val LEnc={len(val_lenc_idx)}  "
    f"Val Ransomware={len(val_ran_idx)}  Conti_06={len(c06_idx)}")

models = {}
for name, score_file, thresh_file in [
    ("all-classes (vae_fixed)", "vae_fixed_window_scores.npy", "vae_fixed_threshold.npy"),
    ("non-ran-only (vae_nonran)", "vae_nonran_window_scores.npy", "vae_nonran_threshold.npy"),
]:
    sp = FEATURES_DIR / score_file
    tp = FEATURES_DIR / thresh_file
    if not sp.exists() or not tp.exists():
        log(f"  WARNING: missing {sp} or {tp} — skipping {name}")
        continue
    scores = np.load(sp)
    thresh = float(np.load(tp).flat[0])
    models[name] = (scores, thresh)
    log(f"  Loaded {name}: scores shape={scores.shape}, threshold={thresh:.4f}")

json_out = {}

log(); log("=" * 72); log("PER-CLASS SCORE DISTRIBUTION — SIDE BY SIDE"); log("=" * 72)

for name, (scores, thresh) in models.items():
    log(f"\n  ── {name} (threshold={thresh:.4f}) ──")
    log(f"  {'Class':<16}  {'N':>4}  {'Mean':>8}  {'P50':>8}  {'P95':>8}  {'>thresh':>8}")
    model_stats = {}
    for cls_name, idx_arr in [
        ("Val Benign", val_ben_idx), ("Val LEnc", val_lenc_idx),
        ("Val Ransomware", val_ran_idx), ("Conti_06", np.array(c06_idx)),
    ]:
        sc = scores[idx_arr]
        stats = {
            "n": len(sc), "mean": float(sc.mean()), "p50": float(np.percentile(sc, 50)),
            "p95": float(np.percentile(sc, 95)), "above_thresh": float((sc > thresh).mean()),
        }
        model_stats[cls_name] = stats
        log(f"  {cls_name:<16}  {stats['n']:>4}  {stats['mean']:>8.4f}  "
            f"{stats['p50']:>8.4f}  {stats['p95']:>8.4f}  {stats['above_thresh']:>8.1%}")
    json_out[name] = model_stats

log(); log("=" * 72); log("CONTI_06 PER-EXEC MSE — ALL-CLASSES vs NON-RAN-ONLY"); log("=" * 72)

if "non-ran-only (vae_nonran)" in models:
    nonran_scores, nonran_thresh = models["non-ran-only (vae_nonran)"]
    log(f"\n  {'Idx':>6}  {'AllClass_MSE':>13}  {'NonRan_MSE':>11}  {'Delta':>8}  {'NonRan>thr':>10}")
    log("  " + "─" * 56)
    c06_compare = []
    for i in c06_idx:
        ac = REF_ALLCLASS_C06.get(i, float("nan"))
        nr = float(nonran_scores[i])
        delta = nr - ac
        fires = nr > nonran_thresh
        log(f"  {i:>6}  {ac:>13.4f}  {nr:>11.4f}  {delta:>+8.4f}  {'YES' if fires else 'no':>10}")
        c06_compare.append({"idx": i, "all_classes_mse": ac, "nonran_mse": nr, "delta": delta, "nonran_fires": fires})
    json_out["c06_compare"] = c06_compare

    mean_abs_delta = np.mean([abs(c["delta"]) for c in c06_compare])
    log(f"\n  Mean |delta| across the 10 Conti_06 execs: {mean_abs_delta:.4f}")
else:
    log("  vae_nonran scores not available — skipping comparison.")

log()
log("=" * 72)
log("READ: if Val Ransomware mean MSE and per-exec Conti_06 values are close")
log("between the two models, training-set-independence is confirmed — the")
log("VAE's reconstruction error in this feature space does not depend much")
log("on what it was trained to reconstruct.")
log("=" * 72)

out_txt  = FEATURES_DIR / "diagnostic_score_dist_results.txt"
out_json = FEATURES_DIR / "diagnostic_score_dist_results.json"
out_txt.write_text("\n".join(lines))
with open(out_json, "w") as f:
    json.dump(json_out, f, indent=2)
log(f"\nSaved: {out_txt}")
log(f"Saved: {out_json}")
log("diagnostic_score_dist.py done.")
