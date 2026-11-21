"""
D2b — mpnet Re-embedding + D2 Oracle for mpnet Representation
==============================================================
Per HANDOFF update 33 (Opus decision, July 2 2026) — runs PARALLEL with D3.

Step 1: Re-extract syscall text windows from the same raw CSV/zip logs that
        generated text_sequences.npy, then embed with all-mpnet-base-v2 (768-d).
        Saves: mpnet_sequences.npy  shape (1970, 300, 768)

Step 2: Run the D2 separability oracle for mpnet (Conti_06 LOO-CV + KNN).
        Compares against already-known results:
          MiniLM (384-d)  — multi-class LOO 10%, KNN ran-frac 6%
          feat_23 (69-d)  — multi-class LOO 40%, KNN ran-frac 42%
        If mpnet adds over MiniLM: embedding dimension was the bottleneck → swap embedder.
        If feat_23 still wins: text pipeline discards entropy trajectory → hybrid (D5).

Decision map (for mpnet only; MiniLM/feat_23 results from original D2 run):
  mpnet multi-class LOO > 50%  → mpnet separable; embedding bottleneck confirmed
  mpnet multi-class LOO < 20%  → mpnet no better; feat_23 hybrid (D5) is the path
  20–50%                       → mpnet marginal; report alongside feat_23 result

★ DIAGNOSTIC ONLY — mpnet oracle uses Conti_06 labels in training for THIS oracle only.
   Never quoted as zero-day detection. Not part of deployed detector.

Saves:
  mpnet_sequences.npy          — (1970, 300, 768) mpnet embeddings
  diag_d2b_mpnet_oracle.txt    — D2 oracle results for mpnet only
  diag_d2b_mpnet_oracle.json   — structured results
"""

import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
MPNET_ID     = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM    = 768
BATCH_SIZE   = 512

RANSOMWARE_CLASSES   = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES    = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES       = ["Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + LEGIT_ENC_CLASSES + BENIGN_CLASSES
VARIANT_CLASSES      = [f"Conti_{i:02d}" for i in range(1, 8)]

lines = []
def log(s=""): print(s, flush=True); lines.append(s)

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
        return super().default(o)


# ════════════════════════════════════════════════════════════════════════════════
# STEP 0 — Check if mpnet_sequences.npy already exists; skip embedding if so
# ════════════════════════════════════════════════════════════════════════════════
mpnet_path = FEATURES_DIR / "mpnet_sequences.npy"

log("=" * 72)
log("D2b — Step 1: mpnet embedding")
log("=" * 72)

if mpnet_path.exists():
    log(f"mpnet_sequences.npy already exists ({mpnet_path.stat().st_size / 1e9:.2f} GB). Loading...")
    mpnet_seqs = np.load(mpnet_path)
    log(f"Loaded shape: {mpnet_seqs.shape}")
    assert mpnet_seqs.shape[1:] == (300, EMBED_DIM), \
        f"Unexpected shape {mpnet_seqs.shape} — expected (N, 300, 768)"
    EMBEDDED = True
else:
    EMBEDDED = False


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Text extraction (same logic as text_embedding.py)
# ════════════════════════════════════════════════════════════════════════════════
T_WINDOW_NS = int(0.1 * 1_000_000_000)   # 100 ms in nanoseconds
T_D_NS      = int(30.0 * 1_000_000_000)  # 30 s in nanoseconds

def load_csv_or_zip(exec_dir: Path, stem: str) -> pd.DataFrame:
    csv_path = exec_dir / f"{stem}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:    return pd.read_csv(csv_path, header=None)
        except: pass
    zip_path = exec_dir / f"{stem}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                inner = f"{stem}.csv" if f"{stem}.csv" in names else (names[0] if names else None)
                if inner:
                    with zf.open(inner) as f:
                        return pd.read_csv(f, header=None)
        except: pass
    return pd.DataFrame()

def log_lines_to_text(ata_write_lines, mem_write_lines):
    tokens = []
    if ata_write_lines.empty:
        tokens.append("no storage write activity")
    else:
        for _, row in ata_write_lines.iterrows():
            size    = int(row.iloc[3]) if not pd.isna(row.iloc[3]) else 0
            entropy = float(row.iloc[4]) if not pd.isna(row.iloc[4]) else -1.0
            ent_label  = ("unknown" if entropy < 0 else
                          "very-high-entropy" if entropy > 0.9 else
                          "high-entropy"      if entropy > 0.7 else
                          "moderate-entropy"  if entropy > 0.4 else
                          "low-entropy"       if entropy > 0.1 else "near-zero-entropy")
            size_label = ("large-write" if size > 65536 else
                          "medium-write" if size > 4096 else "small-write")
            tokens.append(f"storage-write {size_label} {ent_label}")
    if mem_write_lines.empty:
        tokens.append("no memory write activity")
    else:
        for _, row in mem_write_lines.iterrows():
            entropy  = float(row.iloc[4]) if not pd.isna(row.iloc[4]) else 0.0
            mem_type = int(row.iloc[5])   if not pd.isna(row.iloc[5]) else 1
            type_label = {1: "4kb-page", 2: "2mb-page", 4: "mmio"}.get(mem_type, "unknown-page")
            ent_label  = ("very-high-entropy" if entropy > 0.9 else
                          "high-entropy"      if entropy > 0.7 else
                          "moderate-entropy"  if entropy > 0.4 else
                          "low-entropy"       if entropy > 0.1 else "near-zero-entropy")
            tokens.append(f"memory-write {type_label} {ent_label}")
    tokens = tokens[:73]
    return " . ".join(tokens)

def extract_text_windows(exec_dir):
    exec_dir = Path(exec_dir)
    ata_w = load_csv_or_zip(exec_dir, "ata_write")
    mem_w = load_csv_or_zip(exec_dir, "mem_write")
    ref_ns = None
    if not ata_w.empty:
        ref_ns = int(ata_w.iloc[0, 0]) * 1_000_000_000 + int(ata_w.iloc[0, 1])
    if not mem_w.empty:
        t = int(mem_w.iloc[0, 0]) * 1_000_000_000 + int(mem_w.iloc[0, 1])
        ref_ns = t if ref_ns is None else min(ref_ns, t)
    if ref_ns is None:
        return ["no activity"] * 300
    if not ata_w.empty:
        ata_rel = (ata_w.iloc[:, 0].astype(np.int64) * 1_000_000_000 +
                   ata_w.iloc[:, 1].astype(np.int64)) - ref_ns
        ata_w = ata_w.copy(); ata_w["rel_ns"] = ata_rel
    if not mem_w.empty:
        mem_rel = (mem_w.iloc[:, 0].astype(np.int64) * 1_000_000_000 +
                   mem_w.iloc[:, 1].astype(np.int64)) - ref_ns
        mem_w = mem_w.copy(); mem_w["rel_ns"] = mem_rel
    windows = []
    t = 0
    while t <= T_D_NS - T_WINDOW_NS:
        t_end = t + T_WINDOW_NS
        w_ata = ata_w[((ata_w["rel_ns"] >= t) & (ata_w["rel_ns"] < t_end))] if not ata_w.empty else pd.DataFrame()
        w_mem = mem_w[((mem_w["rel_ns"] >= t) & (mem_w["rel_ns"] < t_end))] if not mem_w.empty else pd.DataFrame()
        windows.append(log_lines_to_text(w_ata, w_mem))
        t += T_WINDOW_NS
    return windows


if not EMBEDDED:
    log("\nLoading metadata to find exec_dir paths...")
    meta_df = pd.read_csv(FEATURES_DIR / "metadata.csv")
    n_exec  = len(meta_df)
    log(f"  {n_exec} executions found in metadata.csv")

    # Check a sample exec_dir exists
    sample_dir = Path(meta_df.iloc[0]["exec_dir"])
    if not sample_dir.exists():
        log(f"ERROR: exec_dir not found: {sample_dir}")
        log("Raw CSV logs are not available at the expected path.")
        log("Cannot create mpnet_sequences.npy without raw logs.")
        log("Check: ls $(dirname metadata.csv)  and look for original data directories.")
        raise FileNotFoundError(f"exec_dir not found: {sample_dir}")

    log(f"  exec_dir sample OK: {sample_dir}")

    # Extract text for all executions
    log("\nExtracting text windows from raw CSV/zip logs...")
    all_windows = []
    n_empty = 0
    for i, row in meta_df.iterrows():
        windows = extract_text_windows(row["exec_dir"])
        all_windows.append(windows)
        if windows[0] == "no activity": n_empty += 1
        if (i + 1) % 100 == 0:
            log(f"  Text extraction: {i+1}/{n_exec}  (empty so far: {n_empty})")

    log(f"Text extraction complete. {n_exec} executions, {n_empty} with no activity.")

    # Embed with mpnet
    log(f"\nLoading mpnet model: {MPNET_ID} ...")
    st_model = SentenceTransformer(MPNET_ID)
    log(f"  Embedding dim: {st_model.get_sentence_embedding_dimension()}")

    flat_windows = [w for exec_windows in all_windows for w in exec_windows]
    log(f"  Total windows: {len(flat_windows):,}  (= {n_exec} × 300)")
    log(f"  Batch size: {BATCH_SIZE}")
    log(f"  Estimated batches: {len(flat_windows) // BATCH_SIZE + 1}")
    log(f"  Embedding...")

    flat_embeddings = st_model.encode(
        flat_windows,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    mpnet_seqs = flat_embeddings.reshape(n_exec, 300, EMBED_DIM).astype(np.float32)
    log(f"  mpnet_seqs shape: {mpnet_seqs.shape}  dtype: {mpnet_seqs.dtype}")

    np.save(mpnet_path, mpnet_seqs)
    log(f"\nSaved: {mpnet_path}  ({mpnet_path.stat().st_size / 1e9:.2f} GB)")
    EMBEDDED = True


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — D2 Oracle for mpnet representation
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("D2b — Step 2: Separability oracle for mpnet (768-d)")
log("=" * 72)
log("⚠  Conti_06 labels used in training below — DIAGNOSTIC ONLY.")
log("   Never quoted as zero-day detection rates.")
log()

# Load splits and labels (same as D2)
all_labels = np.load(FEATURES_DIR / "text_labels.npy")
split      = np.load(FEATURES_DIR / "fewshot_split_cnn.npz", allow_pickle=True)
val_idx    = split["val_idx"]
variant_idx = np.where(split["variant_mask"])[0]

train_mask  = np.array([l in ALL_ORIGINAL_CLASSES for l in all_labels])
train_idx   = np.where(train_mask)[0]
c06_idx     = np.where(all_labels == "Conti_06")[0]
val_nonran  = np.array([i for i in val_idx if all_labels[i] not in RANSOMWARE_CLASSES])

def label_to_3class(lbl):
    if lbl in RANSOMWARE_CLASSES: return 2
    if lbl in LEGIT_ENC_CLASSES:  return 1
    return 0

train_y = np.array([label_to_3class(all_labels[i]) for i in train_idx])
c06_y   = np.full(len(c06_idx), 2)

log(f"  Training pool: {len(train_idx)}  |  Conti_06: {len(c06_idx)}  |  Val non-ran: {len(val_nonran)}")

# mpnet pooled: mean over 300 windows → (N, 768)
mpnet_pooled = mpnet_seqs.mean(axis=1)   # (1970, 768)
log(f"  mpnet pooled shape: {mpnet_pooled.shape}")


def loo_oracle(X_all, repr_name):
    """Multi-class LOO-CV over 10 Conti_06 executions."""
    n_c06 = len(c06_idx)
    records = []
    for fold in range(n_c06):
        test_i     = c06_idx[fold]
        train_c06  = np.delete(c06_idx, fold)
        tr_idx_all = np.concatenate([train_idx, train_c06])
        tr_y_all   = np.concatenate([train_y, np.full(len(train_c06), 2)])
        X_tr = X_all[tr_idx_all]; X_te = X_all[[test_i]]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X_tr_s, tr_y_all)
        pred    = clf.predict(X_te_s)[0]
        proba   = clf.predict_proba(X_te_s)[0]
        prob_ran = proba[list(clf.classes_).index(2)] if 2 in clf.classes_ else 0.0
        records.append({"fold": fold, "exec_idx": int(test_i),
                        "pred": int(pred), "prob_ran": float(prob_ran), "correct": bool(pred == 2)})
    det_rate = sum(r["correct"] for r in records) / n_c06
    return records, det_rate


def binary_loo(X_all):
    """Binary LOO: Conti_06 (1) vs val non-ransomware (0)."""
    n_c06 = len(c06_idx)
    records = []
    for fold in range(n_c06):
        test_i    = c06_idx[fold]
        train_c06 = np.delete(c06_idx, fold)
        X_tr = np.concatenate([X_all[train_c06], X_all[val_nonran]])
        y_tr = np.concatenate([np.ones(len(train_c06)), np.zeros(len(val_nonran))])
        X_te = X_all[[test_i]]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(X_tr_s, y_tr)
        pred     = int(clf.predict(X_te_s)[0])
        prob_pos = float(clf.predict_proba(X_te_s)[0][1])
        records.append({"fold": fold, "exec_idx": int(test_i),
                        "pred": pred, "prob_pos": prob_pos, "correct": bool(pred == 1)})
    det_rate = sum(r["correct"] for r in records) / n_c06
    return records, det_rate


def knn_ransomware_frac(X_all, k=5):
    """KNN: fraction of k=5 nearest training neighbors that are ransomware."""
    X_tr = X_all[train_idx]
    sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_c06_s = sc.transform(X_all[c06_idx])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_tr_s)
    _, indices = nn.kneighbors(X_c06_s)
    records = []
    for i, inds in enumerate(indices):
        nb_y = train_y[inds]
        records.append({
            "exec_idx": int(c06_idx[i]),
            "ran_frac":  float((nb_y == 2).mean()),
            "ben_frac":  float((nb_y == 0).mean()),
            "lenc_frac": float((nb_y == 1).mean()),
        })
    return records


# Run oracle
log("\n[mpnet pooled (768-d)]")
recs_mc,  det_mc  = loo_oracle(mpnet_pooled, "mpnet")
recs_bin, det_bin = binary_loo(mpnet_pooled)
knn_recs          = knn_ransomware_frac(mpnet_pooled)

CLASS_NAMES = ["Benign", "Legit-Enc", "Ransomware"]
log(f"\n  Multi-class LOO (3-class, Conti_06 vs known classes):")
log(f"  {'Exec':>6}  {'Pred':>10}  {'P(ran)':>8}  {'Correct?':>10}")
log(f"  {'─'*40}")
for r in recs_mc:
    flag = "✅" if r["correct"] else "✗ "
    pred_name = CLASS_NAMES[r["pred"]] if r["pred"] < 3 else "Unknown"
    log(f"  {r['exec_idx']:>6}  {pred_name:>10}  {r['prob_ran']:>8.3f}  {flag}")
log(f"  Multi-class LOO detection rate: {det_mc:.0%}  ({sum(r['correct'] for r in recs_mc)}/10)")

log(f"\n  Binary LOO (Conti_06 vs val non-ransomware):")
log(f"  {'Exec':>6}  {'Pred':>10}  {'P(C06)':>8}  {'Correct?':>10}")
log(f"  {'─'*40}")
for r in recs_bin:
    flag = "✅" if r["correct"] else "✗ "
    log(f"  {r['exec_idx']:>6}  {'C06' if r['pred']==1 else 'Non-ran':>10}  {r['prob_pos']:>8.3f}  {flag}")
log(f"  Binary LOO detection rate: {det_bin:.0%}  ({sum(r['correct'] for r in recs_bin)}/10)")

avg_ran = np.mean([r["ran_frac"] for r in knn_recs])
avg_ben = np.mean([r["ben_frac"] for r in knn_recs])
log(f"\n  KNN (k=5 nearest training neighbors, cosine):")
log(f"  Mean ransomware fraction: {avg_ran:.1%}  |  Mean benign fraction: {avg_ben:.1%}")
log(f"  {'Exec':>6}  {'Ran%':>6}  {'Ben%':>6}  {'LEnc%':>6}")
for r in knn_recs:
    log(f"  {r['exec_idx']:>6}  {r['ran_frac']:>6.0%}  {r['ben_frac']:>6.0%}  {r['lenc_frac']:>6.0%}")


# ════════════════════════════════════════════════════════════════════════════════
# Decision
# ════════════════════════════════════════════════════════════════════════════════
log()
log("=" * 72)
log("D2b DECISION MAP")
log("=" * 72)
log()
log("  Known D2 results (from original run):")
log("  MiniLM (384-d) : multi-class LOO 10%,  KNN ran-frac 6%")
log("  feat_23 (69-d) : multi-class LOO 40%,  KNN ran-frac 42%")
log()
log(f"  mpnet (768-d)  : multi-class LOO {det_mc:.0%},  KNN ran-frac {avg_ran:.0%}  ← THIS RUN")
log()

if det_mc > 0.50 and det_mc > 0.40:  # mpnet beats feat_23
    verdict = "mpnet_best"
    log("  ★ mpnet > feat_23 AND separable:")
    log("    → Embedding dimension was the bottleneck.")
    log("    → Consider switching primary embedder from MiniLM to mpnet.")
    log("    → D5 hybrid still useful but mpnet alone may be sufficient.")
elif det_mc > 0.50:  # mpnet separable but not better than feat_23
    verdict = "mpnet_separable_feat23_better"
    log("  ★ mpnet separable (>50%) but feat_23 still better:")
    log("    → Text pipeline at 768-d partially recovers the signal.")
    log("    → feat_23 has more discriminative power — D5 hybrid is primary path.")
    log("    → mpnet may add marginal signal as a secondary head.")
elif det_mc > 0.20:
    verdict = "mpnet_marginal"
    log("  ★ mpnet marginal (20-50%) — does not clearly separate Conti_06.")
    log("    → Embedding dimension alone is NOT the bottleneck.")
    log("    → feat_23 signal (40%) is more discriminative than mpnet text.")
    log("    → D5 hybrid (feat_23 head alongside text classifier) is the primary path.")
else:
    verdict = "mpnet_not_separable"
    log("  ★ mpnet NOT separable (<20%) — no better than MiniLM.")
    log("    → Neither text representation separates Conti_06.")
    log("    → feat_23 behavioral features are the ONLY source of signal.")
    log("    → D5 hybrid (feat_23 discriminative head) is the required path.")
    log("    → This strengthens the D3 LSTM/LR head result as the key experiment.")


# ── PCA visualization ─────────────────────────────────────────────────────────
log("\nGenerating PCA visualization for mpnet...")
from sklearn.decomposition import PCA

val_benign_idx = [i for i in val_idx if all_labels[i] in BENIGN_CLASSES]
val_lenc_idx   = [i for i in val_idx if all_labels[i] in LEGIT_ENC_CLASSES]
val_ran_idx    = [i for i in val_idx if all_labels[i] in RANSOMWARE_CLASSES]
plot_idx       = np.array(val_benign_idx + val_lenc_idx + val_ran_idx + list(c06_idx))
plot_labels    = all_labels[plot_idx]

color_map = {
    **{l: "steelblue" for l in BENIGN_CLASSES},
    **{l: "orange"    for l in LEGIT_ENC_CLASSES},
    **{l: "crimson"   for l in RANSOMWARE_CLASSES},
    "Conti_06": "lime",
}
colors = [color_map.get(l, "gray") for l in plot_labels]
is_c06 = np.array([l == "Conti_06" for l in plot_labels])

fig, ax = plt.subplots(figsize=(8, 6))
X_plot  = mpnet_pooled[plot_idx]
pca     = PCA(n_components=2, random_state=42)
coords  = pca.fit_transform(StandardScaler().fit_transform(X_plot))

ax.scatter(coords[~is_c06, 0], coords[~is_c06, 1],
           c=[c for c, m in zip(colors, is_c06) if not m],
           alpha=0.3, s=10, linewidths=0)
ax.scatter(coords[is_c06, 0], coords[is_c06, 1],
           c="lime", edgecolors="black", s=150, marker="*",
           linewidths=1, zorder=5, label="Conti_06")

ax.set_title(f"mpnet (768-d) PCA\nConti_06 ★ vs val classes | multi-class LOO: {det_mc:.0%}", fontsize=10)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

from matplotlib.patches import Patch
legend_patches = [
    Patch(color="steelblue", label="Benign (val)"),
    Patch(color="orange",    label="Legit-Enc (val)"),
    Patch(color="crimson",   label="Ransomware (val)"),
    Patch(color="lime",      label="Conti_06"),
]
ax.legend(handles=legend_patches, fontsize=8)
plt.tight_layout()
pca_path = FEATURES_DIR / "diag_d2b_mpnet_pca.png"
plt.savefig(pca_path, dpi=150)
log(f"Saved PCA: {pca_path}")


# ── Save outputs ──────────────────────────────────────────────────────────────
json_out = {
    "mpnet_pooled_shape": list(mpnet_seqs.shape),
    "multiclass_loo": {"det_rate": det_mc, "records": recs_mc},
    "binary_loo":     {"det_rate": det_bin, "records": recs_bin},
    "knn_mean_ran_frac": float(avg_ran),
    "knn_mean_ben_frac": float(avg_ben),
    "knn_records": knn_recs,
    "verdict": verdict,
    "known_d2_reference": {
        "MiniLM_384d": {"multiclass_loo": 0.10, "knn_ran_frac": 0.06},
        "feat23_69d":  {"multiclass_loo": 0.40, "knn_ran_frac": 0.42},
    },
}

summary = "\n".join(lines)
(FEATURES_DIR / "diag_d2b_mpnet_oracle.txt").write_text(summary)
with open(FEATURES_DIR / "diag_d2b_mpnet_oracle.json", "w") as f:
    json.dump(json_out, f, indent=2, cls=NpEncoder)

print(f"\nSaved: diag_d2b_mpnet_oracle.txt / .json / diag_d2b_mpnet_pca.png")
print("D2b done.")
