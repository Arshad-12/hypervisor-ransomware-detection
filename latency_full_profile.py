"""
latency_full_profile.py — Complete Latency Profile: All 4 Stages, Mean/Median/P95
================================================================================================
Per HANDOFF item 66 (professor comment). Supersedes LATENCY_BENCHMARK (July 11).

The comment: "Report mean, median, and 95th-percentile latency, including feature extraction,
embedding, anomaly scoring, and fusion."

Gaps in the July benchmark: (1) only mean ± std was kept, no per-rep timings, so no
median/P95 could be computed; (2) only 2 of her 4 stages were measured (embedding, anomaly
scoring) — feature extraction (raw CSVs -> (300,23), including per-4KiB Shannon entropy over
30s of I/O) and fusion (the threshold decision) were never measured at all. Feature extraction
is the one most likely to dominate, since the currently-reported 0.82ms decision-time figure
excludes it entirely.

★ Streaming REMOVED — per item 63 (proven to contribute exactly nothing, an implementation bug).
★ Every figure below is explicitly labelled per-execution or per-window — the July table's
  "per 0.1s window" label on what were actually per-execution forward passes caused a real
  error in the report and must not recur.
★ Hardware reported as a single H100 GPU (feature extraction itself is CPU/pandas-bound I/O,
  timed on the same node for a consistent environment).

Stages measured, each as ONE well-defined per-execution operation, >=100 repetitions each,
so mean/median/P95 are computed directly (not extrapolated from per-window numbers, which
would mis-state percentiles under summation):
  1. Feature extraction  — extract_features(exec_dir): raw CSVs -> (300,23). Timed on >=100
     REAL executions sampled from metadata.csv (genuine I/O + pandas parsing, not synthetic).
  2. Embedding            — MiniLM over a full 300-line log (one execution's worth), batched.
  3. Anomaly scoring      — 2-way CNN forward (1,300,384) + anomaly head forward (1,300,23),
     reported both combined and broken out individually.
  4. Fusion               — the threshold decision rule itself (a few float comparisons).
End-to-end total reported as the SUM of per-stage statistics (standard latency-budgeting
convention — mean of sums = sum of means exactly; P95-of-sum is approximated conservatively
by the sum of per-stage P95s, stated explicitly as an upper-bound convention, not a separately
jointly-measured quantity).
"""

import time, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sentence_transformers import SentenceTransformer

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
NREPS_FEATEXTRACT = 100
NREPS_EMBED       = 150
NREPS_SCORE       = 200
NREPS_FUSION       = 500
WARMUP = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

log(f"Device: {device}")
if device.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)} (single H100)")

def stats_ms(ts_list, label, unit="per execution"):
    arr = np.array(ts_list)
    mean_ms = float(arr.mean()); median_ms = float(np.percentile(arr, 50)); p95_ms = float(np.percentile(arr, 95))
    log(f"  {label:<40} [{unit}]  mean={mean_ms:8.3f}ms  median={median_ms:8.3f}ms  P95={p95_ms:8.3f}ms  (n={len(arr)})")
    return {"mean": mean_ms, "median": median_ms, "p95": p95_ms, "n": len(arr), "unit": unit}


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Feature extraction (raw CSVs -> (300,23)), real executions, CPU/IO
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("STAGE 1 — Feature extraction (raw CSVs -> (300,23) array)"); log("=" * 72)
log("  Reference implementation: reextract_feat23.py (exact copy below). Timed on REAL")
log("  executions' raw CSVs, not synthetic data. Includes per-4KiB Shannon entropy etc.")

T_WINDOW, T_D, EPSILON = 0.1, 30.0, 0.1
TYPE_4KB, TYPE_2MB, TYPE_MMIO = 1, 2, 4

def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0: return pd.DataFrame()
    try: return pd.read_csv(path, header=None)
    except Exception: return pd.DataFrame()

def load_csv_or_zip(exec_dir: Path, stem: str) -> pd.DataFrame:
    csv_path = exec_dir / f"{stem}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try: return pd.read_csv(csv_path, header=None)
        except Exception: pass
    zip_path = exec_dir / f"{stem}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                inner = f"{stem}.csv" if f"{stem}.csv" in names else (names[0] if names else None)
                if inner:
                    with zf.open(inner) as f: return pd.read_csv(f, header=None)
        except Exception: pass
    return pd.DataFrame()

def calc_throughput(lines_df):
    if lines_df.empty: return 0.0
    return float(lines_df.iloc[:, 3].sum()) / T_WINDOW

def calc_LBA_variance(lines_df):
    if len(lines_df) < 2: return 0.0
    return float(lines_df.iloc[:, 2].astype(np.float64).var(ddof=1))

def calc_shannon_entropy(lines_df):
    if lines_df.empty: return 0.0
    valid = lines_df.iloc[:, 4]; valid = valid[valid >= 0.0]
    return float(valid.mean()) if not valid.empty else 0.0

def calculate_storage_feature(exec_dir: Path):
    lines_sr = load_csv_or_zip(exec_dir, "ata_read")
    lines_sw = load_csv_or_zip(exec_dir, "ata_write")
    if lines_sr.empty and lines_sw.empty: return np.zeros((0, 5), dtype=np.float32)
    ref_ns = None
    if not lines_sr.empty:
        t = lines_sr.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sr.iloc[:, 1].astype(np.int64)
        ref_ns = t.iloc[0]
    if not lines_sw.empty:
        t = lines_sw.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sw.iloc[:, 1].astype(np.int64)
        if ref_ns is None or t.iloc[0] < ref_ns: ref_ns = t.iloc[0]
    if not lines_sr.empty:
        t = lines_sr.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sr.iloc[:, 1].astype(np.int64)
        lines_sr = lines_sr.copy(); lines_sr["rel_ns"] = t - ref_ns
    if not lines_sw.empty:
        t = lines_sw.iloc[:, 0].astype(np.int64) * 1_000_000_000 + lines_sw.iloc[:, 1].astype(np.int64)
        lines_sw = lines_sw.copy(); lines_sw["rel_ns"] = t - ref_ns
    T_window_ns = int(T_WINDOW * 1e9); T_d_ns = int(T_D * 1e9); epsilon_ns = int(EPSILON * 1e9)
    chi_s = []; t = 0
    while t <= T_d_ns - T_window_ns:
        t_end = t + T_window_ns
        w_sr = lines_sr[(lines_sr["rel_ns"] >= t) & (lines_sr["rel_ns"] < t_end)] if not lines_sr.empty else pd.DataFrame()
        w_sw = lines_sw[(lines_sw["rel_ns"] >= t) & (lines_sw["rel_ns"] < t_end)] if not lines_sw.empty else pd.DataFrame()
        chi_s.append([calc_throughput(w_sr), calc_throughput(w_sw), calc_LBA_variance(w_sr),
                      calc_LBA_variance(w_sw), calc_shannon_entropy(w_sw)])
        t += epsilon_ns
    return np.array(chi_s, dtype=np.float32)

def count_page_access(lines_df, page_type):
    if lines_df.empty: return 0
    return int((lines_df.iloc[:, 5] == page_type).sum())

def calc_GPA_variance(lines_df):
    if len(lines_df) < 2: return 0.0
    return float(lines_df.iloc[:, 2].astype(np.float64).var(ddof=1))

def calc_memory_entropy(lines_df):
    if lines_df.empty: return 0.0
    entropy_col = lines_df.iloc[:, 4]; valid = entropy_col[entropy_col > 0.0]
    return float(valid.mean()) if not valid.empty else 0.0

def calculate_memory_feature(exec_dir: Path):
    lines_mr  = load_csv_safe(exec_dir / "mem_read.csv")
    lines_mw  = load_csv_safe(exec_dir / "mem_write.csv")
    lines_mrw = load_csv_safe(exec_dir / "mem_readwrite.csv")
    lines_mx  = load_csv_safe(exec_dir / "mem_exec.csv")
    all_frames = [f for f in [lines_mr, lines_mw, lines_mrw, lines_mx] if not f.empty]
    if not all_frames: return np.zeros((0, 18), dtype=np.float32)
    ref_ns = None
    for df in all_frames:
        t = df.iloc[:, 0].astype(np.int64) * 1_000_000_000 + df.iloc[:, 1].astype(np.int64)
        t0 = t.iloc[0]
        if ref_ns is None or t0 < ref_ns: ref_ns = t0
    frames_with_rel = []
    for df in [lines_mr, lines_mw, lines_mrw, lines_mx]:
        if not df.empty:
            t = df.iloc[:, 0].astype(np.int64) * 1_000_000_000 + df.iloc[:, 1].astype(np.int64)
            df = df.copy(); df["rel_ns"] = t - ref_ns
        frames_with_rel.append(df)
    lines_mr, lines_mw, lines_mrw, lines_mx = frames_with_rel
    T_window_ns = int(T_WINDOW * 1e9); T_d_ns = int(T_D * 1e9); epsilon_ns = int(EPSILON * 1e9)
    chi_m = []; t = 0
    while t <= T_d_ns - T_window_ns:
        t_end = t + T_window_ns
        def window(df):
            if df.empty: return df
            return df[(df["rel_ns"] >= t) & (df["rel_ns"] < t_end)]
        w_mr, w_mw, w_mrw, w_mx = window(lines_mr), window(lines_mw), window(lines_mrw), window(lines_mx)
        H_mw, H_mrw = calc_memory_entropy(w_mw), calc_memory_entropy(w_mrw)
        chi_m.append([
            H_mw, H_mrw,
            count_page_access(w_mr, TYPE_4KB), count_page_access(w_mw, TYPE_4KB),
            count_page_access(w_mrw, TYPE_4KB), count_page_access(w_mx, TYPE_4KB),
            count_page_access(w_mr, TYPE_2MB), count_page_access(w_mw, TYPE_2MB),
            count_page_access(w_mrw, TYPE_2MB), count_page_access(w_mx, TYPE_2MB),
            count_page_access(w_mr, TYPE_MMIO), count_page_access(w_mw, TYPE_MMIO),
            count_page_access(w_mrw, TYPE_MMIO), count_page_access(w_mx, TYPE_MMIO),
            calc_GPA_variance(w_mr), calc_GPA_variance(w_mw), calc_GPA_variance(w_mrw), calc_GPA_variance(w_mx),
        ])
        t += epsilon_ns
    return np.array(chi_m, dtype=np.float32)

def extract_features(exec_dir: Path):
    chi_s = calculate_storage_feature(exec_dir)
    chi_m = calculate_memory_feature(exec_dir)
    if chi_s.shape[0] == 0 or chi_m.shape[0] == 0: return None
    n = min(chi_s.shape[0], chi_m.shape[0])
    return np.concatenate([chi_s[:n], chi_m[:n]], axis=1)

meta_df = pd.read_csv(FEATURES_DIR / "metadata.csv")
sample_execs = meta_df.sample(min(NREPS_FEATEXTRACT, len(meta_df)), random_state=42)
log(f"  Timing feature extraction on {len(sample_execs)} real executions...")
featextract_ts = []
n_fail = 0
for _, row in sample_execs.iterrows():
    exec_dir = Path(row["exec_dir"])
    t0 = time.perf_counter()
    result = extract_features(exec_dir)
    t1 = time.perf_counter()
    if result is not None:
        featextract_ts.append((t1 - t0) * 1000)
    else:
        n_fail += 1
if n_fail:
    log(f"  ({n_fail} executions failed to extract — excluded from timing, likely missing local CSVs)")
if len(featextract_ts) < 10:
    log(f"  WARNING: only {len(featextract_ts)} successful timings — raw CSVs may not be present on "
        f"this node (they live under dataset/, not /scratch/aabdulj4/RanSMAP/features/). Falling back "
        f"is not performed; report whatever succeeded.")
stage1 = stats_ms(featextract_ts, "Feature extraction (raw CSV -> feat_23)", "per execution") if featextract_ts else None


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Embedding (MiniLM over a full 300-line execution log, batched)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("STAGE 2 — Embedding (MiniLM, one execution's 300 log lines, batched)"); log("=" * 72)

log("  Loading MiniLM (sentence-transformers/all-MiniLM-L6-v2)...")
t0 = time.time()
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
log(f"  Loaded in {time.time()-t0:.2f}s")

dummy_lines = [
    "ProcessCreate: explorer.exe -> cmd.exe; FileCreate: C:\\Users\\user\\document.pdf; "
    "RegistrySet: HKCU\\Software\\App\\key=value"
] * 300  # one execution = 300 log lines (0.1s windows over 30s)

@torch.no_grad()
def fn_embed_execution():
    sbert.encode(dummy_lines, convert_to_tensor=True, show_progress_bar=False, batch_size=300)

for _ in range(WARMUP): fn_embed_execution()
if device.type == "cuda": torch.cuda.synchronize()
embed_ts = []
for _ in range(NREPS_EMBED):
    t0 = time.perf_counter(); fn_embed_execution()
    if device.type == "cuda": torch.cuda.synchronize()
    embed_ts.append((time.perf_counter() - t0) * 1000)
stage2 = stats_ms(embed_ts, "Embedding (batched, 300 lines = 1 execution)", "per execution")
log(f"  Derived per-window (embedding_mean / 300): {stage2['mean']/300:.4f} ms/window "
    f"— informational only, NOT separately measured; collection-time embedding runs "
    f"continuously per 0.1s window and is off the decision-time critical path.")


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Anomaly scoring (2-way CNN + anomaly head, per execution)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("STAGE 3 — Anomaly scoring (2-way CNN forward + anomaly head forward)"); log("=" * 72)

class TextEmbeddingClassifier2Way(nn.Module):
    def __init__(self, emb_dim=384, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.classifier    = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_classes))
        self.reconstructor = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, emb_dim))
    def forward(self, x):
        feat = self.conv(x.permute(0, 2, 1)); pooled = feat.mean(dim=-1)
        return self.classifier(pooled), self.reconstructor(pooled), pooled

class BiLSTMClassifier(nn.Module):
    """Architecture only — weights don't affect latency, so random init is fine for timing."""
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

clf2 = TextEmbeddingClassifier2Way(emb_dim=384, n_classes=2).to(device)
clf2.load_state_dict(torch.load(FEATURES_DIR / "classifier_2way.pt", map_location=device))
clf2.eval()
log("  classifier_2way.pt loaded (real trained weights).")

bilstm = BiLSTMClassifier().to(device)
bilstm.eval()
log("  Anomaly head: architecture-matched, randomly-initialized (weights don't affect latency).")

seq_300_384 = torch.randn(1, 300, 384, dtype=torch.float32).to(device)
seq_300_23  = torch.randn(1, 300, 23, dtype=torch.float32).to(device)

@torch.no_grad()
def fn_cnn(): clf2(seq_300_384)
@torch.no_grad()
def fn_head(): bilstm(seq_300_23)
@torch.no_grad()
def fn_scoring_combined(): clf2(seq_300_384); bilstm(seq_300_23)

for fn in [fn_cnn, fn_head, fn_scoring_combined]:
    for _ in range(WARMUP): fn()
if device.type == "cuda": torch.cuda.synchronize()

cnn_ts, head_ts, combined_ts = [], [], []
for _ in range(NREPS_SCORE):
    t0 = time.perf_counter(); fn_cnn()
    if device.type == "cuda": torch.cuda.synchronize()
    cnn_ts.append((time.perf_counter() - t0) * 1000)
for _ in range(NREPS_SCORE):
    t0 = time.perf_counter(); fn_head()
    if device.type == "cuda": torch.cuda.synchronize()
    head_ts.append((time.perf_counter() - t0) * 1000)
for _ in range(NREPS_SCORE):
    t0 = time.perf_counter(); fn_scoring_combined()
    if device.type == "cuda": torch.cuda.synchronize()
    combined_ts.append((time.perf_counter() - t0) * 1000)

stats_ms(cnn_ts, "  2-way CNN forward alone", "per execution")
stats_ms(head_ts, "  Anomaly head forward alone", "per execution")
stage3 = stats_ms(combined_ts, "Anomaly scoring (CNN + head combined)", "per execution")


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Fusion (the threshold decision rule)
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("STAGE 4 — Fusion (threshold decision, streaming removed per item 63)"); log("=" * 72)

def fusion_rule(p_ran, head_score, thr):
    if p_ran >= 0.5: return "Attack"
    if head_score > thr: return "Attack"
    return "Benign"

fusion_ts = []
for _ in range(WARMUP): fusion_rule(0.3, 0.1, 0.5)
for _ in range(NREPS_FUSION):
    t0 = time.perf_counter()
    fusion_rule(0.3, 0.1, 0.5)
    fusion_ts.append((time.perf_counter() - t0) * 1000)
stage4 = stats_ms(fusion_ts, "Fusion (threshold decision rule)", "per execution")


# ════════════════════════════════════════════════════════════════════════════════
# END-TO-END TOTAL
# ════════════════════════════════════════════════════════════════════════════════
log(); log("=" * 72); log("END-TO-END TOTAL (sum of per-stage statistics)"); log("=" * 72)
log("  Convention: total MEAN = sum of stage means (exact). Total P95 = sum of stage P95s")
log("  (a standard conservative latency-budgeting convention, not a separately jointly-")
log("  measured quantity — the true P95-of-sum is generally tighter than this).")

stages = {"Feature extraction": stage1, "Embedding": stage2, "Anomaly scoring": stage3, "Fusion": stage4}
valid_stages = {k: v for k, v in stages.items() if v is not None}

log(f"\n  {'Stage':<24} {'Mean (ms)':>12} {'Median (ms)':>14} {'P95 (ms)':>12}")
log("  " + "-" * 64)
for name, s in valid_stages.items():
    log(f"  {name:<24} {s['mean']:>12.3f} {s['median']:>14.3f} {s['p95']:>12.3f}")
total_mean   = sum(s["mean"] for s in valid_stages.values())
total_median = sum(s["median"] for s in valid_stages.values())
total_p95    = sum(s["p95"] for s in valid_stages.values())
log("  " + "-" * 64)
log(f"  {'END-TO-END TOTAL':<24} {total_mean:>12.3f} {total_median:>14.3f} {total_p95:>12.3f}")

log()
if stage1 is not None:
    feat_frac = stage1["mean"] / total_mean
    log(f"  Feature extraction is {feat_frac:.1%} of the end-to-end mean total.")
    if feat_frac > 0.5:
        log(f"  -> Feature extraction DOMINATES. The sub-millisecond inference claim covers only")
        log(f"     anomaly scoring + fusion; the true end-to-end cost is bounded by extraction.")
        log(f"     Still cheap against a 30s observation window, but a different sentence than")
        log(f"     'sub-millisecond real-time' for the full pipeline.")
    else:
        log(f"  -> Feature extraction does not dominate; the sub-millisecond real-time claim")
        log(f"     holds reasonably end-to-end.")
log()
log(f"  All figures above are PER EXECUTION (one 30s / 300-window decision), except the")
log(f"  informational per-window embedding rate noted in Stage 2, which is explicitly derived,")
log(f"  not separately measured, and runs continuously off the decision-time critical path.")

out_txt  = FEATURES_DIR / "latency_full_profile_results.txt"
(out_txt).write_text("\n".join(lines))
log(f"\nSaved: {out_txt}")
log("latency_full_profile.py done.")
