"""
D2 — Supervised Separability Oracle for Conti_06
==================================================
DIAGNOSTIC ONLY — Conti_06 labels are used IN TRAINING here.
These results are NEVER part of the deployed detector or any zero-day claim.
Purpose: identify WHICH representation contains separating signal.

Method:
  LOO-CV over 10 Conti_06 executions.
  Each fold: hold out 1 Conti_06 exec; train on [9 Conti_06 + 1,800 training pool];
  ask: does the oracle say "ransomware" for the held-out exec?
  Also: binary classifier (Conti_06 vs val non-ransomware) for direct separability.

Representations tested:
  R1 — MiniLM pooled   : mean of 300 × 384-d window embeddings → 384-d (current)
  R2 — feat_23 stats   : mean + std + max over 300 windows → 69-d (entropy trajectory)
  R3 — mpnet pooled    : mean of 300 × 768-d mpnet embeddings → 768-d
                         (only if mpnet_sequences.npy exists on cluster)

Decision map (per update 32):
  R3 separable, R1 not → embedding dimension bottleneck (mpnet > MiniLM)
  R2 separable, R1/R3 not → text step discards entropy trajectory → go hybrid (D5)
  None separable even supervised → hard information-limit evidence → revert to (31)

Saves:
  diag_d2_separability.txt   — full report + decision map
  diag_d2_separability.json  — structured results
  diag_d2_pca.png            — PCA visualisation per representation
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]
CLASS_TO_INT         = {"benign": 0, "legit_enc": 1, "ransomware": 2}

lines = []
def log(s=""): print(s, flush=True); lines.append(s)


# ── Load metadata and splits ───────────────────────────────────────────────────

log("Loading metadata and splits...")
meta_df = pd.read_csv(FEATURES_DIR / "metadata.csv")
all_labels = np.load(FEATURES_DIR / "text_labels.npy")

split       = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx     = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]
c06_idx     = np.where(all_labels == "Conti_06")[0]
val_nonran  = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])

log(f"  Training pool : {len(train_idx)}")
log(f"  Conti_06      : {len(c06_idx)} executions (all held out from training)")
log(f"  Val non-ran   : {len(val_nonran)}")
log(f"  All variants  : {len(variant_idx)}")


# ── Load features ──────────────────────────────────────────────────────────────

# R1 — MiniLM pooled (mean over 300 windows)
log("\n[R1] Loading MiniLM embeddings (text_sequences.npy)...")
text_seqs = np.load(FEATURES_DIR / "text_sequences.npy")   # (1970, 300, 384)
miniLM_pooled = text_seqs.mean(axis=1)                      # (1970, 384)
log(f"     MiniLM pooled shape: {miniLM_pooled.shape}")

# R2 — feat_23 stats (mean + std + max over 300 windows → 69-d)
log("[R2] Loading feat_23 statistics...")
feat_23 = np.zeros((len(meta_df), 300, 23), dtype=np.float32)
for i, row in meta_df.iterrows():
    feat_23[i] = np.load(row["filepath"])

feat_23_f64  = feat_23.astype(np.float64)          # promote to float64 to avoid overflow
feat23_mean  = feat_23_f64.mean(axis=1)            # (1970, 23)
feat23_std   = feat_23_f64.std(axis=1)             # (1970, 23)
feat23_max   = feat_23_f64.max(axis=1)             # (1970, 23) — captures peaks
feat23_stats = np.concatenate([feat23_mean, feat23_std, feat23_max], axis=1).astype(np.float32)  # (1970, 69)
log(f"     feat_23 stats shape : {feat23_stats.shape}")

# R3 — mpnet pooled (if pre-computed file exists)
mpnet_path = FEATURES_DIR / "mpnet_sequences.npy"
if mpnet_path.exists():
    log("[R3] Loading mpnet embeddings (mpnet_sequences.npy)...")
    mpnet_seqs   = np.load(mpnet_path)       # expected (1970, 300, 768)
    mpnet_pooled = mpnet_seqs.mean(axis=1)   # (1970, 768)
    log(f"     mpnet pooled shape : {mpnet_pooled.shape}")
    HAS_MPNET = True
else:
    log("[R3] mpnet_sequences.npy NOT FOUND — mpnet representation skipped.")
    log("     To enable: run mpnet embedding job and save as mpnet_sequences.npy")
    HAS_MPNET = False
    mpnet_pooled = None

# Build representations dict
representations = {"MiniLM (384-d)": miniLM_pooled, "feat_23 stats (69-d)": feat23_stats}
if HAS_MPNET:
    representations["mpnet (768-d)"] = mpnet_pooled

# Numeric class labels for the training pool
def label_to_int(lbl):
    if lbl in RANSOMWARE_CLASSES or lbl.startswith("Conti_"): return 2  # ransomware
    if lbl in LEGIT_ENC_CLASSES: return 1
    return 0

train_y = np.array([label_to_int(all_labels[i]) for i in train_idx])
c06_y   = np.full(len(c06_idx), 2)  # all Conti_06 are ransomware (oracle)


# ── LOO-CV helper ─────────────────────────────────────────────────────────────

def loo_oracle(X_all, train_idx, train_y, c06_idx, c06_y,
               repr_name, class_weight="balanced"):
    """
    LOO-CV over 10 Conti_06 executions.
    Each fold: train on [training pool + 9 Conti_06] → predict 1 held-out Conti_06.
    Returns: list of (true=2, predicted, prob_ransomware) per fold.
    """
    n_c06 = len(c06_idx)
    records = []

    for fold in range(n_c06):
        test_i   = c06_idx[fold]
        train_c06 = np.delete(c06_idx, fold)   # 9 other Conti_06 execs

        # Training set: pool + 9 Conti_06
        tr_idx  = np.concatenate([train_idx, train_c06])
        tr_y    = np.concatenate([train_y, np.full(len(train_c06), 2)])

        X_tr  = X_all[tr_idx]
        y_tr  = tr_y
        X_te  = X_all[[test_i]]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = LogisticRegression(
            class_weight=class_weight, max_iter=2000,
            C=1.0, solver="lbfgs",
        )
        clf.fit(X_tr_s, y_tr)
        pred   = clf.predict(X_te_s)[0]
        proba  = clf.predict_proba(X_te_s)[0]
        prob_ran = proba[2]   # P(ransomware)

        records.append({
            "fold":     fold,
            "exec_idx": int(test_i),
            "pred":     int(pred),
            "prob_ran": float(prob_ran),
            "correct":  bool(pred == 2),
        })

    det_rate = sum(r["correct"] for r in records) / n_c06
    return records, det_rate


def binary_loo(X_all, val_nonran, c06_idx):
    """
    Binary LOO: Conti_06 (pos=1) vs val non-ransomware (neg=0).
    Simpler setup; directly asks: is Conti_06 separable from benign?
    """
    n_c06 = len(c06_idx)
    records = []

    for fold in range(n_c06):
        test_i    = c06_idx[fold]
        train_c06 = np.delete(c06_idx, fold)

        X_pos = X_all[train_c06]          # 9 Conti_06 (positive)
        X_neg = X_all[val_nonran]         # 180 val non-ransomware (negative)
        X_tr  = np.concatenate([X_pos, X_neg])
        y_tr  = np.concatenate([np.ones(len(train_c06)), np.zeros(len(val_nonran))])

        X_te  = X_all[[test_i]]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(X_tr_s, y_tr)
        pred    = int(clf.predict(X_te_s)[0])
        prob_pos = float(clf.predict_proba(X_te_s)[0][1])

        records.append({
            "fold": fold, "exec_idx": int(test_i),
            "pred": pred, "prob_pos": prob_pos, "correct": bool(pred == 1),
        })

    det_rate = sum(r["correct"] for r in records) / n_c06
    return records, det_rate


# ── KNN distance analysis ─────────────────────────────────────────────────────

def knn_distances(X_all, train_idx, train_y, c06_idx, k=5):
    """
    For each Conti_06 exec, find k nearest neighbors in the training pool.
    Reports: fraction of top-k neighbors that are ransomware vs benign.
    """
    X_tr = X_all[train_idx]
    sc   = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_c06_s = sc.transform(X_all[c06_idx])

    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_tr_s)
    dists, indices = nn.kneighbors(X_c06_s)

    records = []
    for i, (ds, inds) in enumerate(zip(dists, indices)):
        neighbor_labels = train_y[inds]
        ran_frac  = (neighbor_labels == 2).mean()
        ben_frac  = (neighbor_labels == 0).mean()
        lenc_frac = (neighbor_labels == 1).mean()
        records.append({
            "exec_idx":  int(c06_idx[i]),
            "nearest_dist": float(ds[0]),
            "ransomware_frac": float(ran_frac),
            "benign_frac":     float(ben_frac),
            "legit_enc_frac":  float(lenc_frac),
        })
    return records


# ── Run all diagnostics ───────────────────────────────────────────────────────

log("\n" + "=" * 70)
log("D2 — SUPERVISED SEPARABILITY ORACLE (diagnostic only)")
log("=" * 70)
log("⚠  Conti_06 labels used in training below — DIAGNOSTIC ONLY.")
log("   These numbers are NEVER quoted as zero-day detection rates.")
log("=" * 70)

json_out = {"representations": {}}

for repr_name, X_all in representations.items():
    log(f"\n{'─'*70}")
    log(f"  Representation: {repr_name}")
    log(f"{'─'*70}")

    # Multi-class LOO
    records_mc, det_mc = loo_oracle(X_all, train_idx, train_y, c06_idx, c06_y, repr_name)
    log(f"\n  Multi-class LOO (ransomware/legit-enc/benign):")
    log(f"  {'Exec':>6}  {'Pred':>10}  {'P(ran)':>8}  {'Correct?':>10}")
    log(f"  {'─'*42}")
    CLASS_NAMES = ["Benign", "Legit-Enc", "Ransomware"]
    for r in records_mc:
        flag = "✅" if r["correct"] else "✗ "
        log(f"  {r['exec_idx']:>6}  {CLASS_NAMES[r['pred']]:>10}  {r['prob_ran']:>8.3f}  {flag}")
    log(f"  Oracle detection rate (multi-class LOO): {det_mc:.0%}  ({sum(r['correct'] for r in records_mc)}/10)")

    # Binary LOO
    records_bin, det_bin = binary_loo(X_all, val_nonran, c06_idx)
    log(f"\n  Binary LOO (Conti_06 vs val non-ransomware):")
    log(f"  {'Exec':>6}  {'Pred':>10}  {'P(C06)':>8}  {'Correct?':>10}")
    log(f"  {'─'*42}")
    for r in records_bin:
        flag = "✅" if r["correct"] else "✗ "
        log(f"  {r['exec_idx']:>6}  {'C06' if r['pred']==1 else 'Non-ran':>10}  {r['prob_pos']:>8.3f}  {flag}")
    log(f"  Binary detection rate: {det_bin:.0%}  ({sum(r['correct'] for r in records_bin)}/10)")

    # KNN distances
    knn_recs = knn_distances(X_all, train_idx, train_y, c06_idx, k=5)
    avg_ran  = np.mean([r["ransomware_frac"] for r in knn_recs])
    avg_ben  = np.mean([r["benign_frac"]     for r in knn_recs])
    log(f"\n  KNN (k=5 nearest training neighbors, cosine):")
    log(f"  Mean ransomware fraction in top-5: {avg_ran:.1%}")
    log(f"  Mean benign fraction      in top-5: {avg_ben:.1%}")
    log(f"  {'Exec':>6}  {'NearDist':>10}  {'Ran%':>6}  {'Ben%':>6}  {'LEnc%':>6}")
    for r in knn_recs:
        log(f"  {r['exec_idx']:>6}  {r['nearest_dist']:>10.4f}  "
            f"{r['ransomware_frac']:>6.0%}  {r['benign_frac']:>6.0%}  {r['legit_enc_frac']:>6.0%}")

    json_out["representations"][repr_name] = {
        "multiclass_loo": {"det_rate": det_mc, "records": records_mc},
        "binary_loo":     {"det_rate": det_bin, "records": records_bin},
        "knn_mean_ran_frac": avg_ran,
        "knn_mean_ben_frac": avg_ben,
        "knn_records": knn_recs,
    }


# ── Decision map ──────────────────────────────────────────────────────────────

log("\n" + "=" * 70)
log("DECISION MAP (per HANDOFF update 32)")
log("=" * 70)

scores = {k: json_out["representations"][k] for k in json_out["representations"]}

def separable(k, thresh=0.5):
    """Returns True if either LOO det rate exceeds thresh."""
    r = scores[k]
    return r["multiclass_loo"]["det_rate"] > thresh or r["binary_loo"]["det_rate"] > thresh

miniLM_sep = separable("MiniLM (384-d)")
feat23_sep  = separable("feat_23 stats (69-d)")
mpnet_sep   = separable("mpnet (768-d)") if HAS_MPNET else None

log(f"\n  Separability threshold: > 50% detection in either LOO variant")
log(f"  MiniLM  (R1): {'SEPARABLE' if miniLM_sep else 'NOT separable'}")
log(f"  feat_23 (R2): {'SEPARABLE' if feat23_sep else 'NOT separable'}")
if HAS_MPNET:
    log(f"  mpnet   (R3): {'SEPARABLE' if mpnet_sep else 'NOT separable'}")
else:
    log(f"  mpnet   (R3): NOT TESTED (mpnet_sequences.npy not found)")

log()
if not feat23_sep and not miniLM_sep and (mpnet_sep is None or not mpnet_sep):
    log("  ★ VERDICT: NOT SEPARABLE in any representation — even with oracle labels.")
    log("    → Hard information-limit evidence. Conti_06 falls within the benign")
    log("      distribution regardless of representation or architecture.")
    log("    → Update (31) applies: characterize as limits-of-behavioral-detection finding.")
    log("    → D3/D4/D5 NOT warranted. Report the scientific finding.")
    verdict = "information_limit"

elif feat23_sep and not miniLM_sep and (mpnet_sep is None or not mpnet_sep):
    log("  ★ VERDICT: Separable in raw feat_23 but NOT in MiniLM text embeddings.")
    log("    → The text embedding step (MiniLM) is DISCARDING the entropy trajectory.")
    log("    → Conti_06 IS detectable from behavioral features, but the CNN which")
    log("      reads text embeddings cannot see it.")
    log("    → GO TO D5: hybrid / numeric-trajectory anomaly head alongside text classifier.")
    log("    → Also run D3/D4 on the VAE using feat_23 more carefully.")
    verdict = "feat23_separable_text_not"

elif HAS_MPNET and mpnet_sep and not miniLM_sep:
    log("  ★ VERDICT: Separable with mpnet (768-d) but NOT with MiniLM (384-d).")
    log("    → Embedding dimension is the bottleneck.")
    log("    → Switch embedder to mpnet if it passes the D1 latency gate.")
    verdict = "mpnet_separable_miniLM_not"

elif miniLM_sep or (HAS_MPNET and mpnet_sep):
    log("  ★ VERDICT: Separable in text embedding space — the representation has the signal.")
    log("    → The architecture (CNN pooling / VAE) is discarding it.")
    log("    → Go to D3 (anomaly score menu) and D4 (encoder sweep).")
    verdict = "text_separable"

else:
    log("  ★ VERDICT: Ambiguous — marginal separability. See per-representation numbers above.")
    verdict = "ambiguous"

json_out["verdict"] = verdict
if not HAS_MPNET:
    json_out["mpnet_note"] = "mpnet_sequences.npy not found; rerun after computing mpnet embeddings"


# ── PCA Visualization ─────────────────────────────────────────────────────────

log("\nGenerating PCA visualisation ...")

# Subset for plotting: val benign + val legit-enc + val ransomware + all Conti_06
val_benign_idx   = [i for i in val_idx if all_labels[i] in BENIGN_CLASSES]
val_lenc_idx     = [i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES]
val_ran_idx      = [i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES]
plot_idx         = np.array(val_benign_idx + val_lenc_idx + val_ran_idx + list(c06_idx))
plot_labels_raw  = all_labels[plot_idx]

color_map = {
    **{l: "steelblue"  for l in BENIGN_CLASSES},
    **{l: "orange"     for l in LEGIT_ENC_CLASSES},
    **{l: "crimson"    for l in RANSOMWARE_CLASSES},
    "Conti_06": "lime",
}
colors = [color_map.get(l, "gray") for l in plot_labels_raw]
is_c06 = np.array([l == "Conti_06" for l in plot_labels_raw])

n_reprs = len(representations)
fig, axes = plt.subplots(1, n_reprs, figsize=(7 * n_reprs, 6))
if n_reprs == 1: axes = [axes]

for ax, (repr_name, X_all) in zip(axes, representations.items()):
    X_plot = X_all[plot_idx]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(StandardScaler().fit_transform(X_plot))

    # Background: val benign/legit-enc/ransomware
    ax.scatter(coords[~is_c06, 0], coords[~is_c06, 1],
               c=[c for c, m in zip(colors, is_c06) if not m],
               alpha=0.3, s=10, linewidths=0)

    # Conti_06 highlighted
    ax.scatter(coords[is_c06, 0], coords[is_c06, 1],
               c="lime", edgecolors="black", s=120, marker="*",
               linewidths=1, zorder=5, label="Conti_06")

    ax.set_title(f"{repr_name}\nPCA (Conti_06 ★ vs val classes)", fontsize=9)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=8)

# Legend patch
from matplotlib.patches import Patch
legend_patches = [
    Patch(color="steelblue", label="Benign (val)"),
    Patch(color="orange",    label="Legit-Enc (val)"),
    Patch(color="crimson",   label="Ransomware (val)"),
    Patch(color="lime",      label="Conti_06"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=8, frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])
pca_path = FEATURES_DIR / "diag_d2_pca.png"
plt.savefig(pca_path, dpi=150)
log(f"Saved PCA: {pca_path}")


# ── Save outputs ───────────────────────────────────────────────────────────────

summary = "\n".join(lines)
(FEATURES_DIR / "diag_d2_separability.txt").write_text(summary)

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        return super().default(o)

with open(FEATURES_DIR / "diag_d2_separability.json", "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)

print(f"\nSaved: diag_d2_separability.txt / .json / diag_d2_pca.png")
print("D2 done.")
