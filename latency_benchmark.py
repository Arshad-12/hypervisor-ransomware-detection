"""
latency_benchmark.py — End-to-End Pipeline Latency Benchmark
=============================================================
Per HANDOFF update (41), task 2.

Measures the FULL pipeline latency per decision point on the H100.
Pipeline: MiniLM embed → 2-way 1D-CNN → streaming → BiLSTM anomaly head.

Deployment context:
  - System collects 1 window per 0.1 s (= one hypervisor log line, text + feat_23)
  - MiniLM embeds text at COLLECTION time (online, per 0.1 s)
  - DECISION is made once per 30 s (300 windows) or more frequently via streaming
  - Streaming runs on the MOST RECENT 50-window slice (every 5 windows = 0.5 s)
  - BiLSTM runs on the 300-window feat_23 batch at decision time

Measured components:
  T1: MiniLM text embed — 1 string → 384-d vector (collection-time cost per window)
  T2: 2-way CNN forward — (1, 300, 384) batch → verdict (decision-time)
  T3: Streaming — 51 CNN passes, each (1, 300, 384) zero-padded (decision-time)
  T4: BiLSTM forward — (1, 300, 23) batch → anomaly score (decision-time)
  T_total: T2 + T3 + T4 (inference cost at decision point, excl. embedding)

Target: total decision-time inference < 100 ms.

Outputs printed to stdout + saved to latency_results.txt.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sentence_transformers import SentenceTransformer

FEATURES_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
NREPS_EMBED   = 200   # repeats for MiniLM embed timing
NREPS_INFER   = 200   # repeats for CNN / BiLSTM timing
NREPS_STREAM  = 100   # repeats for streaming (more expensive)
WARMUP        = 20    # GPU warmup reps (excluded from timing)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lines = []
def log(s=""):
    print(s, flush=True)
    lines.append(str(s))

log(f"Device: {device}")
if device.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Model definitions ──────────────────────────────────────────────────────────
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
        feat   = self.conv(x.permute(0, 2, 1))
        pooled = feat.mean(dim=-1)
        return self.classifier(pooled), self.reconstructor(pooled), pooled

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_features=23, hidden_dim=64, n_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=0.3 if n_layers > 1 else 0)
        D = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(D, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)

# ── Load models ────────────────────────────────────────────────────────────────
log(); log("=" * 70); log("Loading models"); log("=" * 70)

log("  Loading MiniLM (sentence-transformers/all-MiniLM-L6-v2)...")
t0 = time.time()
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
log(f"  Loaded in {time.time()-t0:.2f}s")

log("  Loading 2-way CNN (classifier_2way.pt)...")
clf2 = TextEmbeddingClassifier2Way(emb_dim=384, n_classes=2).to(device)
clf2.load_state_dict(torch.load(FEATURES_DIR / "classifier_2way.pt",
                                map_location=device, weights_only=True))
clf2.eval()

log("  Loading BiLSTM (lstm_model_best.pt or reconstructing from lstm_scores_all)...")
# Check if standalone lstm model is saved
lstm_path = FEATURES_DIR / "lstm_model_best.pt"
if lstm_path.exists():
    bilstm = BiLSTMClassifier().to(device)
    bilstm.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
    bilstm.eval()
    has_bilstm_model = True
    log("  BiLSTM loaded from lstm_model_best.pt")
else:
    has_bilstm_model = False
    log("  lstm_model_best.pt not found — will benchmark with a randomly-initialized BiLSTM")
    log("  (architecture is identical; weights don't affect latency)")
    bilstm = BiLSTMClassifier().to(device)
    bilstm.eval()

# ── Synthetic inputs ───────────────────────────────────────────────────────────
log(); log("=" * 70); log("Preparing synthetic inputs"); log("=" * 70)

# Realistic text: use an actual log-line from the dataset if available, else a dummy
dummy_text = "ProcessCreate: explorer.exe → cmd.exe; FileCreate: C:\\Users\\user\\document.pdf; RegistrySet: HKCU\\Software\\App\\key=value"

# Realistic embeddings: sample from actual data distribution
log("  Loading sample MiniLM embeddings for realistic input...")
minilm_sample = np.load(FEATURES_DIR / "text_sequences.npy", mmap_mode="r")
seq_300_384 = torch.tensor(minilm_sample[0:1], dtype=torch.float32).to(device)  # (1,300,384)
chunk_50_384 = torch.zeros(1, 300, 384, dtype=torch.float32).to(device)         # zero-padded 50-window
chunk_50_384[0, :50] = torch.tensor(minilm_sample[0, :50], dtype=torch.float32).to(device)

seq_300_23 = torch.randn(1, 300, 23, dtype=torch.float32).to(device)
log(f"  Inputs ready: seq (1,300,384), chunk (1,300,384) zero-padded, feat23 (1,300,23)")

# ── Timing helper ──────────────────────────────────────────────────────────────
def time_gpu_fn(fn, nreps, warmup=WARMUP, label=""):
    """Run fn for warmup + nreps, return mean + std in ms."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(nreps):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    mean_ms = np.mean(ts)
    std_ms  = np.std(ts)
    log(f"  {label}: {mean_ms:.3f} ± {std_ms:.3f} ms  (n={nreps})")
    return mean_ms, std_ms

# ── T1: MiniLM embed ───────────────────────────────────────────────────────────
log(); log("=" * 70); log("T1 — MiniLM text embed (1 string → 384-d vector)"); log("=" * 70)
log("  This is the PER-WINDOW collection-time cost.")

@torch.no_grad()
def fn_embed():
    sbert.encode([dummy_text], convert_to_tensor=True, show_progress_bar=False)

t1_mean, t1_std = time_gpu_fn(fn_embed, NREPS_EMBED, label="MiniLM embed (1 string)")

# ── T2: 2-way CNN forward ──────────────────────────────────────────────────────
log(); log("=" * 70); log("T2 — 2-way CNN forward (1, 300, 384) → verdict"); log("=" * 70)

@torch.no_grad()
def fn_cnn():
    clf2(seq_300_384)

t2_mean, t2_std = time_gpu_fn(fn_cnn, NREPS_INFER, label="2-way CNN forward (300 windows)")

# ── T3: Streaming ─────────────────────────────────────────────────────────────
log(); log("=" * 70); log("T3 — Streaming (51 CNN passes, each (1,300,384) zero-padded)"); log("=" * 70)
log("  Sliding window W=50, stride=5, over 300 windows → 51 passes.")
log("  Run at each 5-window tick (every 0.5 s in deployment).")

@torch.no_grad()
def fn_streaming():
    best = 0.0
    for start in range(0, 300 - 50 + 1, 5):
        # In production: only the active 50-window slice changes each tick
        # We zero-pad to 300 (the correct approach for MiniLM CNN)
        x = chunk_50_384  # pre-built zero-padded tensor (representative)
        logits, _, _ = clf2(x)
        p = torch.softmax(logits, dim=1)[0, 1].item()
        if p > best: best = p

t3_mean, t3_std = time_gpu_fn(fn_streaming, NREPS_STREAM, label="Full streaming (51 CNN passes)")

t3_per_pass = t3_mean / 51
log(f"  Per streaming pass: {t3_per_pass:.3f} ms")

# ── T4: BiLSTM forward ────────────────────────────────────────────────────────
log(); log("=" * 70); log("T4 — BiLSTM anomaly head (1, 300, 23) → score"); log("=" * 70)

@torch.no_grad()
def fn_bilstm():
    bilstm(seq_300_23)

t4_mean, t4_std = time_gpu_fn(fn_bilstm, NREPS_INFER, label="BiLSTM forward (300 windows, 23 feat)")

# ── Total decision-time cost ───────────────────────────────────────────────────
log(); log("=" * 70); log("TOTAL LATENCY BREAKDOWN"); log("=" * 70)

t_decision = t2_mean + t3_mean + t4_mean  # at full-window decision point (every 30s)
t_tick     = t2_mean + t3_per_pass        # at each 5-window streaming tick (every 0.5s)

log()
log(f"  Collection-time (per 0.1 s window):")
log(f"    T1 MiniLM embed:          {t1_mean:7.3f} ms")
log()
log(f"  Decision-time (per 300-window batch = every 30 s):")
log(f"    T2 2-way CNN forward:     {t2_mean:7.3f} ms")
log(f"    T3 streaming (51 passes): {t3_mean:7.3f} ms")
log(f"    T4 BiLSTM forward:        {t4_mean:7.3f} ms")
log(f"    ─────────────────────────────────────")
log(f"    Total decision-time:      {t_decision:7.3f} ms  (target < 100 ms)")
log()
log(f"  Streaming-tick (every 5 windows = every 0.5 s):")
log(f"    T2 CNN forward:           {t2_mean:7.3f} ms")
log(f"    T3 single stream pass:    {t3_per_pass:7.3f} ms")
log(f"    Total per-tick:           {t_tick:7.3f} ms")
log()

if t_decision < 100:
    log(f"  ✓ PASS — full decision-time {t_decision:.1f} ms < 100 ms target")
else:
    log(f"  ✗ FAIL — full decision-time {t_decision:.1f} ms EXCEEDS 100 ms target")

log()
log(f"  Note: T1 ({t1_mean:.1f} ms embed) runs CONTINUOUSLY at collection time,")
log(f"  not blocking the decision path. Decision latency = T2+T3+T4 = {t_decision:.1f} ms.")

# ── Save ──────────────────────────────────────────────────────────────────────
(FEATURES_DIR / "latency_results.txt").write_text("\n".join(lines))
log(); log("Saved: latency_results.txt")
log("latency_benchmark.py done.")
