"""
RanSMAP Feature Extraction
===========================
Implements the paper's data preprocessing algorithms from Figures D.1 and D.2.
Produces 23-dimensional feature vectors per time window.

Storage features (5-dim):
    T_s_r  : average read throughput [byte/s]
    T_s_w  : average write throughput [byte/s]
    V_s_r  : variance of read LBAs
    V_s_w  : variance of write LBAs
    H_s_w  : average Shannon entropy of written sectors

Memory features (18-dim):
    H_m_w, H_m_rw                          : avg entropy of write / readwrite pages
    C_4KB_r, C_4KB_w, C_4KB_rw, C_4KB_x   : count of 4 KiB page accesses
    C_2MB_r, C_2MB_w, C_2MB_rw, C_2MB_x   : count of 2 MiB page accesses
    C_MMIO_r, C_MMIO_w, C_MMIO_rw, C_MMIO_x : count of MMIO accesses
    V_m_r, V_m_w, V_m_rw, V_m_x           : variance of accessed GPAs

Parameters (from paper Section 6.1):
    T_window = 0.1 seconds
    T_d      = 30 seconds
    epsilon  = T_window (non-overlapping windows)

Usage:
    python feature_extraction.py

Outputs:
    features/  directory containing one .npy file per execution directory
    Each .npy file has shape (num_windows, 23) and dtype float32
    A metadata CSV: features/metadata.csv with columns [filepath, label, split]
"""

import os
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Root of your copied data
PROJECT_ROOT = Path.home() / "Claude" / "Projects" / "Directed Study" / "Ransomware"
if not (PROJECT_ROOT / "dataset").exists():
    PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "dataset"

# Output directory for extracted features
OUTPUT_DIR = PROJECT_ROOT / "features"

# Paper parameters
T_WINDOW = 0.1   # seconds — optimal from paper
T_D      = 30.0  # seconds — detection time window
EPSILON  = T_WINDOW  # step size (non-overlapping)

# Memory access type codes (from paper Section 2.3)
TYPE_4KB  = 1
TYPE_2MB  = 2
TYPE_MMIO = 4
# Note: TYPE_1GB = 3 is never observed on Windows 10 per paper

# Class definitions
RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
BENIGN_CLASSES     = ["AESCrypt", "Zip", "SDelete", "Firefox", "Office", "Idle"]
ALL_ORIGINAL_CLASSES = RANSOMWARE_CLASSES + BENIGN_CLASSES

# Variant classes (zero-day proxies — never seen during training)
VARIANT_CLASSES = [f"Conti_{i:02d}" for i in range(1, 8)]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv_safe(path: Path) -> pd.DataFrame:
    """
    Load a RanSMAP CSV file.
    Returns empty DataFrame on missing or empty files.
    Columns depend on file type — we name them generically and let callers
    interpret them.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, header=None)
        return df
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return pd.DataFrame()


def load_csv_or_zip(exec_dir: Path, stem: str) -> pd.DataFrame:
    """
    Load a RanSMAP CSV file, trying plain .csv first then .zip.
    Handles the case where large ATA logs are stored zipped
    (common in the full RanSMAP repo to satisfy GitHub's 100 MB/file limit).
    """
    # Try plain CSV first
    csv_path = exec_dir / f"{stem}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            return pd.read_csv(csv_path, header=None)
        except Exception as e:
            print(f"  Warning: could not read {csv_path}: {e}")

    # Fall back to zip
    zip_path = exec_dir / f"{stem}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                # Prefer stem.csv inside the zip; otherwise take the first entry
                inner = f"{stem}.csv" if f"{stem}.csv" in names else (names[0] if names else None)
                if inner is None:
                    return pd.DataFrame()
                with zf.open(inner) as f:
                    return pd.read_csv(f, header=None)
        except Exception as e:
            print(f"  Warning: could not read {zip_path}: {e}")

    return pd.DataFrame()


def to_relative_ns(df: pd.DataFrame) -> pd.Series:
    """
    Convert UNIX timestamp columns (seconds + nanoseconds) to a single
    relative timestamp in nanoseconds, starting from 0 at first access.
    Columns 0 and 1 are t_s and t_ns respectively.
    """
    abs_ns = df.iloc[:, 0].astype(np.int64) * 1_000_000_000 + df.iloc[:, 1].astype(np.int64)
    return abs_ns - abs_ns.iloc[0]


def extract_window(series: pd.Series, t_ns: float, t_end_ns: float) -> pd.Series:
    """Return rows where relative timestamp falls in [t_ns, t_end_ns)."""
    mask = (series >= t_ns) & (series < t_end_ns)
    return mask


# ── Storage Feature Extraction (Figure D.1) ──────────────────────────────────

def calc_throughput(lines: pd.DataFrame) -> float:
    """Average throughput in bytes/s over T_WINDOW seconds."""
    if lines.empty:
        return 0.0
    total_bytes = lines.iloc[:, 3].sum()   # column 3 is 'size'
    return float(total_bytes) / T_WINDOW


def calc_LBA_variance(lines: pd.DataFrame) -> float:
    """Sample variance of LBA values (column 2)."""
    if len(lines) < 2:
        return 0.0
    lba = lines.iloc[:, 2].astype(np.float64)
    return float(lba.var(ddof=1))


def calc_shannon_entropy(lines: pd.DataFrame) -> float:
    """
    Average Shannon entropy of write operations.
    Column 4 is entropy; values of -1 mean not a write — exclude them.
    """
    if lines.empty:
        return 0.0
    entropy_col = lines.iloc[:, 4]
    valid = entropy_col[entropy_col >= 0.0]
    if valid.empty:
        return 0.0
    return float(valid.mean())


def calculate_storage_feature(exec_dir: Path) -> np.ndarray:
    """
    Figure D.1 implementation.
    Returns chi_s of shape (num_windows, 5).
    """
    lines_sr = load_csv_or_zip(exec_dir, "ata_read")
    lines_sw = load_csv_or_zip(exec_dir, "ata_write")

    # Build relative timestamps
    # Use the earliest timestamp across both files as t=0 (per paper)
    if lines_sr.empty and lines_sw.empty:
        return np.zeros((0, 5), dtype=np.float32)

    ref_ns = None
    if not lines_sr.empty:
        t_sr = lines_sr.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sr.iloc[:, 1].astype(np.int64)
        ref_ns = t_sr.iloc[0]
    if not lines_sw.empty:
        t_sw = lines_sw.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sw.iloc[:, 1].astype(np.int64)
        if ref_ns is None or t_sw.iloc[0] < ref_ns:
            ref_ns = t_sw.iloc[0]

    # Add relative timestamp column to each DataFrame
    if not lines_sr.empty:
        t_sr = lines_sr.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sr.iloc[:, 1].astype(np.int64)
        lines_sr = lines_sr.copy()
        lines_sr["rel_ns"] = t_sr - ref_ns
    if not lines_sw.empty:
        t_sw = lines_sw.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sw.iloc[:, 1].astype(np.int64)
        lines_sw = lines_sw.copy()
        lines_sw["rel_ns"] = t_sw - ref_ns

    T_window_ns = int(T_WINDOW * 1_000_000_000)
    T_d_ns      = int(T_D      * 1_000_000_000)
    epsilon_ns  = int(EPSILON  * 1_000_000_000)

    chi_s = []
    t = 0
    while t <= T_d_ns - T_window_ns:
        t_end = t + T_window_ns

        # Extract lines in this window
        if not lines_sr.empty:
            mask_r = (lines_sr["rel_ns"] >= t) & (lines_sr["rel_ns"] < t_end)
            w_sr = lines_sr[mask_r]
        else:
            w_sr = pd.DataFrame()

        if not lines_sw.empty:
            mask_w = (lines_sw["rel_ns"] >= t) & (lines_sw["rel_ns"] < t_end)
            w_sw = lines_sw[mask_w]
        else:
            w_sw = pd.DataFrame()

        T_sr = calc_throughput(w_sr)
        T_sw = calc_throughput(w_sw)
        V_sr = calc_LBA_variance(w_sr)
        V_sw = calc_LBA_variance(w_sw)
        H_sw = calc_shannon_entropy(w_sw)

        chi_s.append([T_sr, T_sw, V_sr, V_sw, H_sw])
        t += epsilon_ns

    return np.array(chi_s, dtype=np.float32)


# ── Memory Feature Extraction (Figure D.2) ───────────────────────────────────

def count_page_access(lines: pd.DataFrame, page_type: int) -> int:
    """Count rows where the 'type' column (index 5) equals page_type.
    NOTE: do NOT use iloc[:, -1] — rel_ns is appended as the last column
    during windowing, so -1 reads nanosecond timestamps, not {1,2,4}.
    """
    if lines.empty:
        return 0
    type_col = lines.iloc[:, 5]   # fixed index: t_s, t_ns, GPA, size, entropy, type
    return int((type_col == page_type).sum())


def calc_GPA_variance(lines: pd.DataFrame) -> float:
    """Sample variance of GPA values (column 2)."""
    if len(lines) < 2:
        return 0.0
    gpa = lines.iloc[:, 2].astype(np.float64)
    return float(gpa.var(ddof=1))


def calc_memory_entropy(lines: pd.DataFrame) -> float:
    """
    Average Shannon entropy of write / readwrite memory pages.
    Column 4 is entropy; -1 values and zeros for MMIO are excluded.
    """
    if lines.empty:
        return 0.0
    entropy_col = lines.iloc[:, 4]
    valid = entropy_col[entropy_col > 0.0]
    if valid.empty:
        return 0.0
    return float(valid.mean())


def calculate_memory_feature(exec_dir: Path) -> np.ndarray:
    """
    Figure D.2 implementation.
    Returns chi_m of shape (num_windows, 18).
    """
    mem_read_path      = exec_dir / "mem_read.csv"
    mem_write_path     = exec_dir / "mem_write.csv"
    mem_readwrite_path = exec_dir / "mem_readwrite.csv"
    mem_exec_path      = exec_dir / "mem_exec.csv"

    lines_mr  = load_csv_safe(mem_read_path)
    lines_mw  = load_csv_safe(mem_write_path)
    lines_mrw = load_csv_safe(mem_readwrite_path)
    lines_mx  = load_csv_safe(mem_exec_path)

    all_frames = [f for f in [lines_mr, lines_mw, lines_mrw, lines_mx] if not f.empty]
    if not all_frames:
        return np.zeros((0, 18), dtype=np.float32)

    # Find global reference timestamp across all 4 files
    ref_ns = None
    for df in all_frames:
        t = df.iloc[:, 0].astype(np.int64) * 1_000_000_000 + df.iloc[:, 1].astype(np.int64)
        t0 = t.iloc[0]
        if ref_ns is None or t0 < ref_ns:
            ref_ns = t0

    # Add relative timestamp to each
    frames_with_rel = []
    for df in [lines_mr, lines_mw, lines_mrw, lines_mx]:
        if not df.empty:
            t = df.iloc[:, 0].astype(np.int64) * 1_000_000_000 + df.iloc[:, 1].astype(np.int64)
            df = df.copy()
            df["rel_ns"] = t - ref_ns
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
            if df.empty:
                return df
            mask = (df["rel_ns"] >= t) & (df["rel_ns"] < t_end)
            return df[mask]

        w_mr  = window(lines_mr)
        w_mw  = window(lines_mw)
        w_mrw = window(lines_mrw)
        w_mx  = window(lines_mx)

        # Shannon entropy (only write and readwrite have valid entropy)
        H_mw  = calc_memory_entropy(w_mw)
        H_mrw = calc_memory_entropy(w_mrw)

        # 4 KiB page counts
        C_4KB_r  = count_page_access(w_mr,  TYPE_4KB)
        C_4KB_w  = count_page_access(w_mw,  TYPE_4KB)
        C_4KB_rw = count_page_access(w_mrw, TYPE_4KB)
        C_4KB_x  = count_page_access(w_mx,  TYPE_4KB)

        # 2 MiB page counts
        C_2MB_r  = count_page_access(w_mr,  TYPE_2MB)
        C_2MB_w  = count_page_access(w_mw,  TYPE_2MB)
        C_2MB_rw = count_page_access(w_mrw, TYPE_2MB)
        C_2MB_x  = count_page_access(w_mx,  TYPE_2MB)

        # MMIO counts
        C_MMIO_r  = count_page_access(w_mr,  TYPE_MMIO)
        C_MMIO_w  = count_page_access(w_mw,  TYPE_MMIO)
        C_MMIO_rw = count_page_access(w_mrw, TYPE_MMIO)
        C_MMIO_x  = count_page_access(w_mx,  TYPE_MMIO)

        # GPA variance
        V_m_r  = calc_GPA_variance(w_mr)
        V_m_w  = calc_GPA_variance(w_mw)
        V_m_rw = calc_GPA_variance(w_mrw)
        V_m_x  = calc_GPA_variance(w_mx)

        feature_vec = [
            H_mw, H_mrw,
            C_4KB_r,  C_4KB_w,  C_4KB_rw,  C_4KB_x,
            C_2MB_r,  C_2MB_w,  C_2MB_rw,  C_2MB_x,
            C_MMIO_r, C_MMIO_w, C_MMIO_rw, C_MMIO_x,
            V_m_r,    V_m_w,    V_m_rw,    V_m_x,
        ]
        chi_m.append(feature_vec)
        t += epsilon_ns

    return np.array(chi_m, dtype=np.float32)


# ── Combined Feature Extraction ───────────────────────────────────────────────

def extract_features(exec_dir: Path) -> np.ndarray:
    """
    Extract the full 23-dimensional feature matrix for one execution directory.
    Returns array of shape (num_windows, 23).
    Concatenates chi_s (5-dim) and chi_m (18-dim) column-wise.
    Returns None if extraction fails or produces empty output.
    """
    chi_s = calculate_storage_feature(exec_dir)
    chi_m = calculate_memory_feature(exec_dir)

    if chi_s.shape[0] == 0 or chi_m.shape[0] == 0:
        return None

    # Both should have the same number of windows since T_d and T_window are fixed
    # In rare cases rounding can differ by 1 — take the minimum
    n = min(chi_s.shape[0], chi_m.shape[0])
    chi = np.concatenate([chi_s[:n], chi_m[:n]], axis=1)  # (n, 23)
    return chi


# ── Dataset Scanning ─────────────────────────────────────────────────────────

def scan_original(data_root: Path) -> list[dict]:
    """
    Scan the original classes across all cpu/ram subdirectories.
    Structure: data_root/original/{cpu}/{ram}/{class}/{exec}
    """
    records = []
    original_root = data_root / "original"
    if not original_root.exists():
        print(f"Warning: original dir not found: {original_root}")
        return records

    for cpu_dir in sorted(original_root.iterdir()):
        if not cpu_dir.is_dir():
            continue
        for ram_dir in sorted(cpu_dir.iterdir()):
            if not ram_dir.is_dir():
                continue
            for class_name in ALL_ORIGINAL_CLASSES:
                class_path = ram_dir / class_name
                if not class_path.exists():
                    continue
                exec_dirs = sorted([d for d in class_path.iterdir() if d.is_dir()])
                for exec_dir in exec_dirs:
                    records.append({
                        "exec_dir": exec_dir,
                        "label": class_name,
                        "group": "original"
                    })
    return records


def scan_variants(data_root: Path) -> list[dict]:
    """
    Scan the 7 Conti variants across all cpu/ram subdirectories.
    Structure: data_root/variants/{cpu}/{ram}/{variant}/{exec}
    """
    records = []
    variants_root = data_root / "variants"
    if not variants_root.exists():
        print(f"Warning: variants dir not found: {variants_root}")
        return records

    for cpu_dir in sorted(variants_root.iterdir()):
        if not cpu_dir.is_dir():
            continue
        for ram_dir in sorted(cpu_dir.iterdir()):
            if not ram_dir.is_dir():
                continue
            for variant_name in VARIANT_CLASSES:
                variant_path = ram_dir / variant_name
                if not variant_path.exists():
                    continue
                exec_dirs = sorted([d for d in variant_path.iterdir() if d.is_dir()])
                for exec_dir in exec_dirs:
                    records.append({
                        "exec_dir": exec_dir,
                        "label": variant_name,
                        "group": "variant"
                    })
    return records


def scan_extra(data_root: Path) -> list[dict]:
    """
    Scan the extra hardware-config executions.
    Structure: data_root/extra/{cpu}/{ram}/{class}/{exec}
    Same classes as original — treated as additional original executions.
    """
    records = []
    extra_root = data_root / "extra"
    if not extra_root.exists():
        return records

    for cpu_dir in sorted(extra_root.iterdir()):
        if not cpu_dir.is_dir():
            continue
        for ram_dir in sorted(cpu_dir.iterdir()):
            if not ram_dir.is_dir():
                continue
            for class_name in ALL_ORIGINAL_CLASSES:
                class_path = ram_dir / class_name
                if not class_path.exists():
                    continue
                for exec_dir in sorted(d for d in class_path.iterdir() if d.is_dir()):
                    records.append({
                        "exec_dir": exec_dir,
                        "label": class_name,
                        "group": "extra"
                    })
    return records


def scan_mix(data_root: Path) -> list[dict]:
    """
    Scan the simultaneous legit-encryption + ransomware executions.
    Structure: data_root/mix/{cpu}/{ram}/{mix_class}/{exec}
    OR (local sample): data_root/{mix_class}/{exec}  (e.g. AESCrypt_Conti/)
    Mix classes are labelled as their own group so downstream code can
    handle them separately (they are NOT used as training labels).
    """
    records = []

    # Full repo: dataset/mix/{cpu}/{ram}/{mix_class}/{exec}
    mix_root = data_root / "mix"
    if mix_root.exists():
        for cpu_dir in sorted(mix_root.iterdir()):
            if not cpu_dir.is_dir():
                continue
            for ram_dir in sorted(cpu_dir.iterdir()):
                if not ram_dir.is_dir():
                    continue
                for mix_class_dir in sorted(ram_dir.iterdir()):
                    if not mix_class_dir.is_dir():
                        continue
                    mix_label = mix_class_dir.name
                    for exec_dir in sorted(d for d in mix_class_dir.iterdir() if d.is_dir()):
                        records.append({
                            "exec_dir": exec_dir,
                            "label": mix_label,
                            "group": "mix"
                        })

    # Local sample fallback: dataset/{AESCrypt_Conti,SDelete_Conti,Zip_Conti}/{exec}
    for mix_label in ["AESCrypt_Conti", "SDelete_Conti", "Zip_Conti"]:
        mix_path = data_root / mix_label
        if mix_path.exists():
            for exec_dir in sorted(d for d in mix_path.iterdir() if d.is_dir()):
                records.append({
                    "exec_dir": exec_dir,
                    "label": mix_label,
                    "group": "mix"
                })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scan all execution directories
    print("Scanning dataset directories...")
    records = scan_original(DATA_ROOT) + scan_variants(DATA_ROOT) + scan_extra(DATA_ROOT) + scan_mix(DATA_ROOT)
    print(f"Found {len(records)} execution directories total.")

    metadata_rows = []
    success = 0
    failed  = 0

    for i, rec in enumerate(records):
        exec_dir = rec["exec_dir"]
        label    = rec["label"]
        group    = rec["group"]

        # Build output filename from relative path
        rel_path = exec_dir.relative_to(DATA_ROOT)
        out_name = str(rel_path).replace("/", "__").replace("\\", "__") + ".npy"
        out_path = OUTPUT_DIR / out_name

        # Skip if already extracted
        if out_path.exists():
            metadata_rows.append({
                "filepath": str(out_path),
                "label": label,
                "group": group,
                "exec_dir": str(exec_dir)
            })
            success += 1
            continue

        print(f"[{i+1}/{len(records)}] {label} / {exec_dir.name}", end=" ... ")

        features = extract_features(exec_dir)

        if features is None:
            print("FAILED (empty output)")
            failed += 1
            continue

        np.save(out_path, features)
        print(f"OK  shape={features.shape}")

        metadata_rows.append({
            "filepath": str(out_path),
            "label": label,
            "group": group,
            "exec_dir": str(exec_dir)
        })
        success += 1

    # Save metadata
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = OUTPUT_DIR / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    print(f"\n{'='*50}")
    print(f"Done. Success: {success}  Failed: {failed}")
    print(f"Features saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {meta_path}")

    # Quick sanity check on one file
    if metadata_rows:
        sample = np.load(metadata_rows[0]["filepath"])
        print(f"\nSanity check — first file: {metadata_rows[0]['label']}")
        print(f"  Shape:  {sample.shape}  (expected ~(300, 23))")
        print(f"  Min:    {sample.min():.4f}")
        print(f"  Max:    {sample.max():.6f}")
        print(f"  Entropy column (col 4, H_s_w) mean: {sample[:, 4].mean():.4f}")


if __name__ == "__main__":
    main()
