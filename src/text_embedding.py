"""
RanSMAP Text Embedding Generation
===================================
Converts raw CSV logs → text window descriptions → MiniLM-L6-v2 embeddings.
Reads metadata from /scratch/aabdulj4/RanSMAP/features/metadata.csv
Writes text_sequences.npy, text_labels.npy, text_3way.npy, text_groups.npy
to /scratch/aabdulj4/RanSMAP/features/
"""

import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
OUTPUT_DIR   = FEATURES_DIR  # same directory

# ── Parameters ────────────────────────────────────────────────────────────────
T_WINDOW    = 0.1
T_D         = 30.0
T_WINDOW_NS = int(T_WINDOW * 1_000_000_000)
T_D_NS      = int(T_D      * 1_000_000_000)

# ── Class definitions ─────────────────────────────────────────────────────────
RANSOMWARE_CLASSES = ["WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti"]
LEGIT_ENC_CLASSES  = ["AESCrypt", "SDelete", "Zip"]
BENIGN_CLASSES     = ["Firefox", "Office", "Idle"]

def get_3way_label(class_name):
    if class_name in RANSOMWARE_CLASSES: return "ransomware"
    if class_name in LEGIT_ENC_CLASSES:  return "legit_encryption"
    if class_name in BENIGN_CLASSES:     return "benign"
    return "unknown"

# ── Text conversion ───────────────────────────────────────────────────────────
def log_lines_to_text(ata_write_lines, mem_write_lines):
    tokens = []

    if ata_write_lines.empty:
        tokens.append("no storage write activity")
    else:
        for _, row in ata_write_lines.iterrows():
            size    = int(row.iloc[3]) if not pd.isna(row.iloc[3]) else 0
            entropy = float(row.iloc[4]) if not pd.isna(row.iloc[4]) else -1.0
            if entropy < 0:       ent_label = "unknown"
            elif entropy > 0.9:   ent_label = "very-high-entropy"
            elif entropy > 0.7:   ent_label = "high-entropy"
            elif entropy > 0.4:   ent_label = "moderate-entropy"
            elif entropy > 0.1:   ent_label = "low-entropy"
            else:                 ent_label = "near-zero-entropy"
            size_label = "large-write" if size > 65536 else \
                         "medium-write" if size > 4096 else "small-write"
            tokens.append(f"storage-write {size_label} {ent_label}")

    if mem_write_lines.empty:
        tokens.append("no memory write activity")
    else:
        for _, row in mem_write_lines.iterrows():
            entropy  = float(row.iloc[4]) if not pd.isna(row.iloc[4]) else 0.0
            mem_type = int(row.iloc[5])   if not pd.isna(row.iloc[5]) else 1
            type_label = {1: "4kb-page", 2: "2mb-page", 4: "mmio"}.get(mem_type, "unknown-page")
            if entropy > 0.9:   ent_label = "very-high-entropy"
            elif entropy > 0.7: ent_label = "high-entropy"
            elif entropy > 0.4: ent_label = "moderate-entropy"
            elif entropy > 0.1: ent_label = "low-entropy"
            else:               ent_label = "near-zero-entropy"
            tokens.append(f"memory-write {type_label} {ent_label}")

    tokens = tokens[:73]
    return " . ".join(tokens)


def load_csv_or_zip(exec_dir: Path, stem: str) -> pd.DataFrame:
    """Load CSV, falling back to .zip if plain .csv is absent or empty."""
    csv_path = exec_dir / f"{stem}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            return pd.read_csv(csv_path, header=None)
        except:
            pass
    zip_path = exec_dir / f"{stem}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                inner = f"{stem}.csv" if f"{stem}.csv" in names else (names[0] if names else None)
                if inner:
                    with zf.open(inner) as f:
                        return pd.read_csv(f, header=None)
        except:
            pass
    return pd.DataFrame()


def extract_text_windows(exec_dir):
    exec_dir = Path(exec_dir)

    ata_w = load_csv_or_zip(exec_dir, "ata_write")
    mem_w = load_csv_or_zip(exec_dir, "mem_write")

    ref_ns = None
    if not ata_w.empty:
        t = int(ata_w.iloc[0, 0]) * 1_000_000_000 + int(ata_w.iloc[0, 1])
        ref_ns = t
    if not mem_w.empty:
        t = int(mem_w.iloc[0, 0]) * 1_000_000_000 + int(mem_w.iloc[0, 1])
        ref_ns = t if ref_ns is None else min(ref_ns, t)

    if ref_ns is None:
        return ["no activity"] * 300

    if not ata_w.empty:
        ata_rel = (ata_w.iloc[:, 0].astype(np.int64) * 1_000_000_000 +
                   ata_w.iloc[:, 1].astype(np.int64)) - ref_ns
        ata_w = ata_w.copy()
        ata_w["rel_ns"] = ata_rel

    if not mem_w.empty:
        mem_rel = (mem_w.iloc[:, 0].astype(np.int64) * 1_000_000_000 +
                   mem_w.iloc[:, 1].astype(np.int64)) - ref_ns
        mem_w = mem_w.copy()
        mem_w["rel_ns"] = mem_rel

    windows = []
    t = 0
    while t <= T_D_NS - T_WINDOW_NS:
        t_end = t + T_WINDOW_NS
        w_ata = pd.DataFrame()
        w_mem = pd.DataFrame()
        if not ata_w.empty:
            mask = (ata_w["rel_ns"] >= t) & (ata_w["rel_ns"] < t_end)
            w_ata = ata_w[mask]
        if not mem_w.empty:
            mask = (mem_w["rel_ns"] >= t) & (mem_w["rel_ns"] < t_end)
            w_mem = mem_w[mask]
        windows.append(log_lines_to_text(w_ata, w_mem))
        t += T_WINDOW_NS

    return windows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    meta_df = pd.read_csv(FEATURES_DIR / "metadata.csv")
    print(f"Processing {len(meta_df)} execution directories...")

    all_windows = []
    all_labels  = []
    all_3way    = []
    all_groups  = []

    for i, row in meta_df.iterrows():
        exec_dir   = Path(row["exec_dir"])
        label      = row["label"]
        group      = row["group"]
        label_3way = get_3way_label(label) if group == "original" else "variant"

        windows = extract_text_windows(exec_dir)
        all_windows.append(windows)
        all_labels.append(label)
        all_3way.append(label_3way)
        all_groups.append(group)

        if (i + 1) % 50 == 0:
            print(f"  Text extraction: {i+1}/{len(meta_df)}")

    all_labels = np.array(all_labels)
    all_3way   = np.array(all_3way)
    all_groups = np.array(all_groups)

    print(f"\nText extraction complete. {len(all_windows)} executions.")
    print(f"3-way distribution: { {l: (all_3way==l).sum() for l in np.unique(all_3way)} }")

    # Embed with MiniLM
    print("\nLoading SentenceTransformer...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_dim  = 384

    flat_windows = [w for exec_windows in all_windows for w in exec_windows]
    print(f"Total windows to embed: {len(flat_windows):,}")
    print("Embedding (this will take a while)...")

    flat_embeddings = st_model.encode(
        flat_windows,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    n_exec    = len(all_windows)
    n_windows = 300
    text_seqs = flat_embeddings.reshape(n_exec, n_windows, emb_dim)

    print(f"\nFinal shape: {text_seqs.shape}")

    np.save(OUTPUT_DIR / "text_sequences.npy", text_seqs)
    np.save(OUTPUT_DIR / "text_labels.npy",    all_labels)
    np.save(OUTPUT_DIR / "text_3way.npy",      all_3way)
    np.save(OUTPUT_DIR / "text_groups.npy",    all_groups)

    print("Saved:")
    print(f"  {OUTPUT_DIR / 'text_sequences.npy'}  {text_seqs.shape}")
    print(f"  {OUTPUT_DIR / 'text_labels.npy'}")
    print(f"  {OUTPUT_DIR / 'text_3way.npy'}")
    print(f"  {OUTPUT_DIR / 'text_groups.npy'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
