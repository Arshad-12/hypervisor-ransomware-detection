"""
RanSMAP feat_23 Re-extraction (Bug Fix)
=========================================
Re-extracts all 1,970 feat_23 .npy files using the FIXED count_page_access
function. Overwrites existing .npy files unconditionally.

Bug fixed: count_page_access previously used iloc[:, -1] which read the
rel_ns (nanosecond timestamp) column instead of the type column at index 5.
This caused C_4KB / C_2MB / C_MMIO features (indices 7-18) to be zero.

Run on cluster:
    sbatch slurm_reextract.sh
"""

import sys
import time
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

# ── Cluster paths ─────────────────────────────────────────────────────────────
FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
META_CSV     = FEATURES_DIR / "metadata.csv"

# ── Paper parameters ──────────────────────────────────────────────────────────
T_WINDOW = 0.1   # seconds
T_D      = 30.0  # seconds
EPSILON  = T_WINDOW

# Memory access type codes
TYPE_4KB  = 1
TYPE_2MB  = 2
TYPE_MMIO = 4


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, header=None)
    except Exception as e:
        print(f"  Warning: {path}: {e}")
        return pd.DataFrame()


def load_csv_or_zip(exec_dir: Path, stem: str) -> pd.DataFrame:
    csv_path = exec_dir / f"{stem}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            return pd.read_csv(csv_path, header=None)
        except Exception as e:
            print(f"  Warning: {csv_path}: {e}")
    zip_path = exec_dir / f"{stem}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                inner = f"{stem}.csv" if f"{stem}.csv" in names else (names[0] if names else None)
                if inner:
                    with zf.open(inner) as f:
                        return pd.read_csv(f, header=None)
        except Exception as e:
            print(f"  Warning: {zip_path}: {e}")
    return pd.DataFrame()


# ── Storage features (Figure D.1) ─────────────────────────────────────────────

def calc_throughput(lines: pd.DataFrame) -> float:
    if lines.empty:
        return 0.0
    return float(lines.iloc[:, 3].sum()) / T_WINDOW


def calc_LBA_variance(lines: pd.DataFrame) -> float:
    if len(lines) < 2:
        return 0.0
    return float(lines.iloc[:, 2].astype(np.float64).var(ddof=1))


def calc_shannon_entropy(lines: pd.DataFrame) -> float:
    if lines.empty:
        return 0.0
    valid = lines.iloc[:, 4]
    valid = valid[valid >= 0.0]
    return float(valid.mean()) if not valid.empty else 0.0


def calculate_storage_feature(exec_dir: Path) -> np.ndarray:
    lines_sr = load_csv_or_zip(exec_dir, "ata_read")
    lines_sw = load_csv_or_zip(exec_dir, "ata_write")

    if lines_sr.empty and lines_sw.empty:
        return np.zeros((0, 5), dtype=np.float32)

    ref_ns = None
    if not lines_sr.empty:
        t = lines_sr.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sr.iloc[:, 1].astype(np.int64)
        ref_ns = t.iloc[0]
    if not lines_sw.empty:
        t = lines_sw.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sw.iloc[:, 1].astype(np.int64)
        if ref_ns is None or t.iloc[0] < ref_ns:
            ref_ns = t.iloc[0]

    if not lines_sr.empty:
        t = lines_sr.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sr.iloc[:, 1].astype(np.int64)
        lines_sr = lines_sr.copy(); lines_sr["rel_ns"] = t - ref_ns
    if not lines_sw.empty:
        t = lines_sw.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sw.iloc[:, 1].astype(np.int64)
        lines_sw = lines_sw.copy(); lines_sw["rel_ns"] = t - ref_ns

    T_window_ns = int(T_WINDOW * 1_000_000_000)
    T_d_ns      = int(T_D      * 1_000_000_000)
    epsilon_ns  = int(EPSILON  * 1_000_000_000)

    chi_s = []
    t = 0
    while t <= T_d_ns - T_window_ns:
        t_end = t + T_window_ns
        w_sr = lines_sr[  (lines_sr["rel_ns"]  >= t) & (lines_sr["rel_ns"]  < t_end)] if not lines_sr.empty  else pd.DataFrame()
        w_sw = lines_sw[  (lines_sw["rel_ns"]  >= t) & (lines_sw["rel_ns"]  < t_end)] if not lines_sw.empty  else pd.DataFrame()
        chi_s.append([
            calc_throughput(w_sr), calc_throughput(w_sw),
            calc_LBA_variance(w_sr), calc_LBA_variance(w_sw),
            calc_shannon_entropy(w_sw),
        ])
        t += epsilon_ns
    return np.array(chi_s, dtype=np.float32)


# ── Memory features (Figure D.2) ──────────────────────────────────────────────

def count_page_access(lines: pd.DataFrame, page_type: int) -> int:
    """Count rows where the type column (index 5) equals page_type.
    FIX: use iloc[:, 5] not iloc[:, -1].
    rel_ns is appended as the LAST column during windowing, so -1 was
    reading nanosecond timestamps (~10^18), never equal to {1, 2, 4}.
    """
    if lines.empty:
        return 0
    type_col = lines.iloc[:, 5]   # fixed index: t_s, t_ns, GPA, size, entropy, type
    return int((type_col == page_type).sum())


def calc_GPA_variance(lines: pd.DataFrame) -> float:
    if len(lines) < 2:
        return 0.0
    return float(lines.iloc[:, 2].astype(np.float64).var(ddof=1))


def calc_memory_entropy(lines: pd.DataFrame) -> float:
    if lines.empty:
        return 0.0
    entropy_col = lines.iloc[:, 4]
    valid = entropy_col[entropy_col > 0.0]
    return float(valid.mean()) if not valid.empty else 0.0


def calculate_memory_feature(exec_dir: Path) -> np.ndarray:
    lines_mr  = load_csv_safe(exec_dir / "mem_read.csv")
    lines_mw  = load_csv_safe(exec_dir / "mem_write.csv")
    lines_mrw = load_csv_safe(exec_dir / "mem_readwrite.csv")
    lines_mx  = load_csv_safe(exec_dir / "mem_exec.csv")

    all_frames = [f for f in [lines_mr, lines_mw, lines_mrw, lines_mx] if not f.empty]
    if not all_frames:
        return np.zeros((0, 18), dtype=np.float32)

    ref_ns = None
    for df in all_frames:
        t = df.iloc[:, 0].astype(np.int64) * 1_000_000_000 + df.iloc[:, 1].astype(np.int64)
        t0 = t.iloc[0]
        if ref_ns is None or t0 < ref_ns:
            ref_ns = t0

    frames_with_rel = []
    for df in [lines_mr, lines_mw, lines_mrw, lines_mx]:
        if not df.empty:
            t = df.iloc[:, 0].astype(np.int64) * 1_000_000_000 + df.iloc[:, 1].astype(np.int64)
            df = df.copy(); df["rel_ns"] = t - ref_ns
        frames_with_rel.append(df)
    lines_mr, lines_mw, lines_mrw, lines_mx = frames_with_rel

    T_window_ns = int(T_WINDOW * 1_000_000_000)
    T_d_ns      = int(T_D      * 1_000_000_000)
    epsilon_ns  = int(EPSILON  * 1_000_000_000)

    chi_m = []
    t = 0
    while t <= T_d_ns - T_window_ns:
        t_end = t + T_window_ns

        def window(df):
            if df.empty: return df
            return df[(df["rel_ns"] >= t) & (df["rel_ns"] < t_end)]

        w_mr  = window(lines_mr)
        w_mw  = window(lines_mw)
        w_mrw = window(lines_mrw)
        w_mx  = window(lines_mx)

        H_mw  = calc_memory_entropy(w_mw)
        H_mrw = calc_memory_entropy(w_mrw)

        C_4KB_r  = count_page_access(w_mr,  TYPE_4KB)
        C_4KB_w  = count_page_access(w_mw,  TYPE_4KB)
        C_4KB_rw = count_page_access(w_mrw, TYPE_4KB)
        C_4KB_x  = count_page_access(w_mx,  TYPE_4KB)

        C_2MB_r  = count_page_access(w_mr,  TYPE_2MB)
        C_2MB_w  = count_page_access(w_mw,  TYPE_2MB)
        C_2MB_rw = count_page_access(w_mrw, TYPE_2MB)
        C_2MB_x  = count_page_access(w_mx,  TYPE_2MB)

        C_MMIO_r  = count_page_access(w_mr,  TYPE_MMIO)
        C_MMIO_w  = count_page_access(w_mw,  TYPE_MMIO)
        C_MMIO_rw = count_page_access(w_mrw, TYPE_MMIO)
        C_MMIO_x  = count_page_access(w_mx,  TYPE_MMIO)

        V_m_r  = calc_GPA_variance(w_mr)
        V_m_w  = calc_GPA_variance(w_mw)
        V_m_rw = calc_GPA_variance(w_mrw)
        V_m_x  = calc_GPA_variance(w_mx)

        chi_m.append([
            H_mw, H_mrw,
            C_4KB_r,  C_4KB_w,  C_4KB_rw,  C_4KB_x,
            C_2MB_r,  C_2MB_w,  C_2MB_rw,  C_2MB_x,
            C_MMIO_r, C_MMIO_w, C_MMIO_rw, C_MMIO_x,
            V_m_r,    V_m_w,    V_m_rw,    V_m_x,
        ])
        t += epsilon_ns

    return np.array(chi_m, dtype=np.float32)


def extract_features(exec_dir: Path):
    chi_s = calculate_storage_feature(exec_dir)
    chi_m = calculate_memory_feature(exec_dir)
    if chi_s.shape[0] == 0 or chi_m.shape[0] == 0:
        return None
    n = min(chi_s.shape[0], chi_m.shape[0])
    return np.concatenate([chi_s[:n], chi_m[:n]], axis=1)  # (n, 23)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== RanSMAP feat_23 Re-extraction (count_page_access fix) ===")
    print(f"Reading metadata from: {META_CSV}")

    meta_df = pd.read_csv(META_CSV)
    n_total = len(meta_df)
    print(f"Executions to re-extract: {n_total}")
    print()

    success = 0
    failed  = 0
    t_start = time.time()

    for i, row in meta_df.iterrows():
        exec_dir   = Path(row["exec_dir"])
        out_path   = Path(row["filepath"])
        label      = row["label"]

        feats = extract_features(exec_dir)

        if feats is None:
            print(f"  [{i+1}/{n_total}] FAILED {label} / {exec_dir.name}")
            failed += 1
            continue

        np.save(out_path, feats)
        success += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            eta     = (n_total - i - 1) / rate
            print(f"  [{i+1}/{n_total}]  {elapsed/60:.1f} min elapsed  ETA {eta/60:.1f} min")

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Re-extraction complete in {elapsed_total/60:.1f} min")
    print(f"  Success: {success}  Failed: {failed}")

    # ── Sanity check: feature audit on 10 random executions ──────────────────
    print("\n=== Feature audit (10 random execs) ===")
    FEAT_NAMES = [
        "T_sr", "T_sw", "V_sr", "V_sw", "H_sw",           # 0-4  storage
        "H_mw", "H_mrw",                                    # 5-6  memory entropy
        "C_4KB_r", "C_4KB_w", "C_4KB_rw", "C_4KB_x",      # 7-10
        "C_2MB_r", "C_2MB_w", "C_2MB_rw", "C_2MB_x",      # 11-14
        "C_MMIO_r","C_MMIO_w","C_MMIO_rw","C_MMIO_x",     # 15-18
        "V_m_r",  "V_m_w",  "V_m_rw",  "V_m_x",           # 19-22
    ]

    sample_rows = meta_df.sample(min(10, len(meta_df)), random_state=99)
    all_feats   = []
    for _, row in sample_rows.iterrows():
        arr = np.load(row["filepath"])  # (300, 23)
        all_feats.append(arr)
    stacked = np.concatenate(all_feats, axis=0)  # (N*300, 23)

    print(f"\n{'Feature':<12}  {'Mean':>12}  {'Max':>12}  {'Non-zero%':>10}")
    print("-" * 52)
    for j, name in enumerate(FEAT_NAMES):
        col = stacked[:, j]
        nz  = 100.0 * (col != 0).mean()
        marker = " ← was 0!" if j in range(7, 19) else ""
        print(f"{name:<12}  {col.mean():>12.4f}  {col.max():>12.4f}  {nz:>9.1f}%{marker}")

    dead = [FEAT_NAMES[j] for j in range(7, 19) if (stacked[:, j] == 0).all()]
    if dead:
        print(f"\nWARNING: still zero after fix: {dead}")
        print("Double-check column index — may differ for mem_read/exec vs mem_write.")
    else:
        print("\nAll page-count features (7-18) are non-zero. Fix confirmed.")


if __name__ == "__main__":
    main()
