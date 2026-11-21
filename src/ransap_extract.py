"""
ransap_extract.py — RanSAP Feature Extraction + MiniLM Text Embedding
======================================================================
Extracts the 5 storage features (chi_s) and MiniLM text embeddings
from RanSAP's ata_read.csv + ata_write.csv per execution.

RanSAP CSV format (per execution directory):
  ata_read.csv:  columns = [time_s, time_ns, LBA, size]
  ata_write.csv: columns = [time_s, time_ns, LBA, size, Entropy#1, Entropy#2]

Storage features (5-dim, matches RanSMAP feat_23 columns 0-4):
  T_s_r:  avg read throughput (bytes/s)
  T_s_w:  avg write throughput (bytes/s)
  V_s_r:  variance of read LBAs
  V_s_w:  variance of write LBAs
  H_s_w:  mean Shannon entropy of written sectors (Entropy#1 column)

Text per window: same token format as RanSMAP diag_d2b_mpnet_embed.py
  "storage-write {size_label} {ent_label} . ... . no memory write activity"
  Note: "no memory write activity" always appended — RanSAP has no memory logs.
  This is identical to RanSMAP storage text but always missing memory tokens.

Window parameters: T_window = 0.1 s, T_d = 30 s, non-overlapping (same as RanSMAP).
Sequences padded/truncated to 300 windows.

Outputs (saved to /scratch/aabdulj4/RanSAP/features/):
  ransap_feat5.npy        — (N, 300, 5) float32, storage features
  ransap_text_seqs.npy    — (N, 300, 384) float32, MiniLM embeddings
  ransap_labels.npy       — (N,) str, family labels
  ransap_metadata.csv     — N rows: exec_dir, label, split(unused)
  ransap_extract_log.txt

Usage:
  Set RANSAP_ROOT below to the directory where you extracted the dataset.
  Run on a GPU node (for MiniLM embedding speed).
"""

import os, time
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

# ── Configuration ─────────────────────────────────────────────────────────────
RANSAP_ROOT   = Path("/scratch/aabdulj4/RanSAP/RanSAP-main")
OUTPUT_DIR    = Path("/scratch/aabdulj4/RanSAP/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Window parameters (must match RanSMAP)
T_WINDOW_S  = 0.1
T_D_S       = 30.0
N_WINDOWS   = 300
T_WINDOW_NS = int(T_WINDOW_S * 1e9)
T_D_NS      = int(T_D_S * 1e9)

# Sector size for throughput calculation (512 bytes/sector)
SECTOR_BYTES = 512

lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Device: {device}")

# ── Discover all execution directories ────────────────────────────────────────
log("=" * 70)
log("Discovering RanSAP execution directories")
log("=" * 70)
log(f"  Root: {RANSAP_ROOT}")

exec_dirs  = []
exec_labels= []

# Label normalization: strip variant suffixes (-largefiles, -w10dirs, -w10dirs-01)
# Structure: dataset/{split}/{os-disk}/{Family[-suffix]}/{exec-dir}/ata_write.csv
# Label = parts[-3] (grandparent of exec dir, parent of exec-dir)
STRIP_SUFFIXES = ["-largefiles", "-w10dirs-01", "-w10dirs"]

def normalize_label(raw):
    for sfx in STRIP_SUFFIXES:
        if raw.endswith(sfx):
            return raw[:-len(sfx)]
    return raw

# Walk the entire tree looking for directories containing ata_write.csv
for root, dirs, files in os.walk(RANSAP_ROOT):
    if "ata_write.csv" in files:
        path  = Path(root)
        raw   = path.parts[-2]   # e.g. "Cerber-w10dirs", "AESCrypt", "Excel"
        label = normalize_label(raw)
        exec_dirs.append(path)
        exec_labels.append(label)

log(f"  Found {len(exec_dirs)} execution directories")
label_counts = {}
for lbl in exec_labels:
    label_counts[lbl] = label_counts.get(lbl, 0) + 1
for lbl, cnt in sorted(label_counts.items()):
    log(f"    {lbl}: {cnt}")

if len(exec_dirs) == 0:
    log("ERROR: No execution directories found.")
    log(f"  Expected ata_write.csv files under {RANSAP_ROOT}")
    log("  Check your directory structure and adjust the discovery logic above.")
    raise SystemExit(1)

# ── Helper: load CSV ──────────────────────────────────────────────────────────
def load_csv(path, stem):
    csv_path = path / f"{stem}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, header=None)
            return df
        except: pass
    return pd.DataFrame()

# ── Storage features (chi_s) — identical to RanSMAP feature_extraction.py ────
def calc_throughput(df, n_windows_total=N_WINDOWS):
    if df.empty: return 0.0
    total_bytes = df.iloc[:, 3].fillna(0).astype(float).sum() * SECTOR_BYTES
    return total_bytes / (n_windows_total * T_WINDOW_S + 1e-9)

def calc_throughput_window(df_window):
    if df_window.empty: return 0.0
    return df_window.iloc[:, 3].fillna(0).astype(float).sum() * SECTOR_BYTES / T_WINDOW_S

def calc_lba_variance(df_window):
    if df_window.empty or len(df_window) < 2: return 0.0
    return float(df_window.iloc[:, 2].fillna(0).astype(float).var())

def calc_shannon_entropy(df_window):
    if df_window.empty: return 0.0
    ent_col = df_window.iloc[:, 4].dropna().astype(float)
    return float(ent_col.mean()) if len(ent_col) > 0 else 0.0

def extract_chi_s(exec_dir):
    """Extract 5 storage features per 0.1s window. Returns (300, 5) float32."""
    ata_r = load_csv(exec_dir, "ata_read")
    ata_w = load_csv(exec_dir, "ata_write")

    # Compute reference time (nanoseconds from first event)
    ref_ns = None
    for df in [ata_r, ata_w]:
        if not df.empty:
            t = int(df.iloc[0, 0]) * 1_000_000_000 + int(df.iloc[0, 1])
            ref_ns = t if ref_ns is None else min(ref_ns, t)
    if ref_ns is None:
        return np.zeros((N_WINDOWS, 5), dtype=np.float32)

    def add_rel(df):
        if df.empty: return df
        df = df.copy()
        df["rel_ns"] = (df.iloc[:, 0].astype(np.int64) * 1_000_000_000 +
                        df.iloc[:, 1].astype(np.int64)) - ref_ns
        return df

    ata_r = add_rel(ata_r)
    ata_w = add_rel(ata_w)

    chi_s = []
    t = 0
    while t < N_WINDOWS * T_WINDOW_NS:
        t_end = t + T_WINDOW_NS
        w_r = ata_r[(ata_r["rel_ns"] >= t) & (ata_r["rel_ns"] < t_end)] if not ata_r.empty else pd.DataFrame()
        w_w = ata_w[(ata_w["rel_ns"] >= t) & (ata_w["rel_ns"] < t_end)] if not ata_w.empty else pd.DataFrame()
        T_sr = calc_throughput_window(w_r)
        T_sw = calc_throughput_window(w_w)
        V_sr = calc_lba_variance(w_r)
        V_sw = calc_lba_variance(w_w)
        H_sw = calc_shannon_entropy(w_w)
        chi_s.append([T_sr, T_sw, V_sr, V_sw, H_sw])
        t += T_WINDOW_NS
        if len(chi_s) >= N_WINDOWS:
            break

    arr = np.array(chi_s, dtype=np.float32)
    # Pad or truncate to exactly N_WINDOWS
    if len(arr) < N_WINDOWS:
        pad = np.zeros((N_WINDOWS - len(arr), 5), dtype=np.float32)
        arr = np.vstack([arr, pad])
    return arr[:N_WINDOWS]

# ── Text extraction (storage-only, same token format as RanSMAP) ──────────────
def storage_row_to_tokens(df_window):
    """Convert one window's ata_write rows to text tokens."""
    tokens = []
    if df_window.empty:
        tokens.append("no storage write activity")
    else:
        for _, row in df_window.iterrows():
            size    = int(row.iloc[3])   if not pd.isna(row.iloc[3]) else 0
            entropy = float(row.iloc[4]) if not pd.isna(row.iloc[4]) else -1.0
            ent_label  = ("unknown"           if entropy < 0     else
                          "very-high-entropy" if entropy > 0.9   else
                          "high-entropy"      if entropy > 0.7   else
                          "moderate-entropy"  if entropy > 0.4   else
                          "low-entropy"       if entropy > 0.1   else "near-zero-entropy")
            size_label = ("large-write"  if size > 65536 else
                          "medium-write" if size > 4096  else "small-write")
            tokens.append(f"storage-write {size_label} {ent_label}")
    # Always add "no memory write activity" — RanSAP has no memory logs
    tokens.append("no memory write activity")
    tokens = tokens[:73]
    return " . ".join(tokens)

def extract_text_windows(exec_dir):
    """Return list of 300 text strings, one per 0.1s window."""
    ata_w = load_csv(exec_dir, "ata_write")
    if ata_w.empty:
        return ["no storage write activity . no memory write activity"] * N_WINDOWS

    ref_ns = int(ata_w.iloc[0, 0]) * 1_000_000_000 + int(ata_w.iloc[0, 1])
    ata_w = ata_w.copy()
    ata_w["rel_ns"] = (ata_w.iloc[:, 0].astype(np.int64) * 1_000_000_000 +
                       ata_w.iloc[:, 1].astype(np.int64)) - ref_ns

    windows = []
    t = 0
    while t < N_WINDOWS * T_WINDOW_NS:
        t_end = t + T_WINDOW_NS
        w_w = ata_w[(ata_w["rel_ns"] >= t) & (ata_w["rel_ns"] < t_end)]
        windows.append(storage_row_to_tokens(w_w))
        t += T_WINDOW_NS
        if len(windows) >= N_WINDOWS:
            break

    while len(windows) < N_WINDOWS:
        windows.append("no storage write activity . no memory write activity")
    return windows[:N_WINDOWS]

# ── Extract all execs ─────────────────────────────────────────────────────────
log()
log("=" * 70)
log("Extracting storage features (chi_s)")
log("=" * 70)

N = len(exec_dirs)
all_feat5  = np.zeros((N, N_WINDOWS, 5), dtype=np.float32)
all_texts  = []   # list of lists of 300 strings

t0 = time.time()
for ii, (exec_dir, label) in enumerate(zip(exec_dirs, exec_labels)):
    all_feat5[ii] = extract_chi_s(exec_dir)
    all_texts.append(extract_text_windows(exec_dir))
    if (ii + 1) % 50 == 0:
        log(f"  {ii+1}/{N} execs done ({time.time()-t0:.0f}s)")

log(f"  All {N} execs extracted in {time.time()-t0:.0f}s")

# ── MiniLM text embedding ─────────────────────────────────────────────────────
log()
log("=" * 70)
log("MiniLM text embedding (all 300 windows × N execs)")
log("=" * 70)

sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
log(f"  Model loaded. Embedding {N} × {N_WINDOWS} = {N * N_WINDOWS} strings...")

all_text_seqs = np.zeros((N, N_WINDOWS, 384), dtype=np.float32)
BATCH = 512  # strings per encode call

t0 = time.time()
for ii in range(N):
    windows = all_texts[ii]
    vecs = []
    for start in range(0, N_WINDOWS, BATCH):
        batch = windows[start:start + BATCH]
        emb   = sbert.encode(batch, convert_to_numpy=True, show_progress_bar=False,
                              normalize_embeddings=False)
        vecs.append(emb.astype(np.float32))
    all_text_seqs[ii] = np.vstack(vecs)
    if (ii + 1) % 50 == 0:
        log(f"  {ii+1}/{N} execs embedded ({time.time()-t0:.0f}s)")

log(f"  Embedding done in {time.time()-t0:.0f}s")

# ── Save ──────────────────────────────────────────────────────────────────────
log()
log("=" * 70)
log("Saving outputs")
log("=" * 70)

labels_arr = np.array(exec_labels)
np.save(OUTPUT_DIR / "ransap_feat5.npy",     all_feat5)
np.save(OUTPUT_DIR / "ransap_text_seqs.npy", all_text_seqs)
np.save(OUTPUT_DIR / "ransap_labels.npy",    labels_arr)

meta_df = pd.DataFrame({"exec_dir": [str(d) for d in exec_dirs], "label": exec_labels})
meta_df.to_csv(OUTPUT_DIR / "ransap_metadata.csv", index=False)

log(f"  ransap_feat5.npy:     {all_feat5.shape}")
log(f"  ransap_text_seqs.npy: {all_text_seqs.shape}")
log(f"  ransap_labels.npy:    {labels_arr.shape}  unique={np.unique(labels_arr).tolist()}")
log(f"  ransap_metadata.csv:  {len(meta_df)} rows")

log(); log("Label distribution:")
for lbl, cnt in sorted(label_counts.items()):
    log(f"  {lbl}: {cnt}")

log()
log("CAVEAT: RanSAP text windows contain storage tokens only +")
log("        'no memory write activity' (no memory logs in RanSAP).")
log("        MiniLM classifier was trained on RanSMAP (storage+memory text).")
log("        Cross-dataset transfer uses storage entropy signal only — partial input.")

(OUTPUT_DIR / "ransap_extract_log.txt").write_text("\n".join(lines))
log(); log("ransap_extract.py done.")
