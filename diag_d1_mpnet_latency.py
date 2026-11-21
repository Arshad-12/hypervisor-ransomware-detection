"""
D1 — mpnet Latency Benchmark on H100
======================================
Context: mpnet (all-mpnet-base-v2, 768-d) was rejected as the text embedder
because it was measured at ~151 ms/window. That measurement was likely on CPU
or without batching. This script confirms the actual GPU latency.

Real-time gate: each 0.1 s window must be embedded in < 100 ms.
Batched-as-deployed gate: entire 30 s execution (300 windows) embedded
in a single batch before inference — no strict per-window constraint;
benchmark total batch throughput.

Models benchmarked:
  MiniLM : sentence-transformers/all-MiniLM-L6-v2   (384-d, currently used)
  mpnet  : sentence-transformers/all-mpnet-base-v2   (768-d, under evaluation)

Saves: diag_d1_mpnet_latency.txt + .json
"""

import json
import time
import numpy as np
import torch
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("sentence_transformers not found — install with: pip install sentence-transformers --break-system-packages")

OUTPUT_DIR = Path("/scratch/aabdulj4/RanSMAP/features")
GATE_MS    = 100.0   # real-time gate per window

# ── Synthetic syscall text ─────────────────────────────────────────────────────
# Modelled on hypervisor-captured syscall sequences: a 0.1 s window typically
# contains 20–120 events formatted as syscall names / arguments.
# We bracket with short, medium, and long texts.
SHORT_TEXT = (
    "NtReadFile NtWriteFile NtCreateFile NtClose NtQueryInformationFile "
    "NtReadFile NtWriteFile NtClose"
)
MEDIUM_TEXT = (
    "NtReadFile NtWriteFile NtCreateFile NtClose NtQueryInformationFile "
    "NtReadFile NtWriteFile NtCreateSection NtMapViewOfSection NtUnmapViewOfSection "
    "NtClose NtReadFile NtWriteFile NtCreateFile NtQueryInformationFile NtClose "
    "NtReadFile NtWriteFile NtCreateSection NtMapViewOfSection NtReadFile NtClose "
    "NtQueryInformationFile NtReadFile NtWriteFile NtClose NtReadFile NtWriteFile"
)
LONG_TEXT = (
    "NtReadFile NtWriteFile NtCreateFile NtClose NtQueryInformationFile "
    "NtReadFile NtWriteFile NtCreateSection NtMapViewOfSection NtUnmapViewOfSection "
    "NtClose NtReadFile NtWriteFile NtCreateFile NtQueryInformationFile NtClose "
    "NtReadFile NtWriteFile NtCreateSection NtMapViewOfSection NtReadFile NtClose "
    "NtQueryInformationFile NtReadFile NtWriteFile NtClose NtReadFile NtWriteFile "
    "NtQuerySystemInformation NtAllocateVirtualMemory NtFreeVirtualMemory NtClose "
    "NtOpenFile NtReadFile NtWriteFile NtClose NtQueryInformationFile NtReadFile "
    "NtDeviceIoControlFile NtCreateFile NtWriteFile NtClose NtReadFile NtWriteFile "
    "NtQueryVolumeInformationFile NtReadFile NtWriteFile NtClose NtOpenFile NtReadFile "
    "NtWriteFile NtFlushBuffersFile NtClose NtQueryInformationFile NtSetInformationFile "
    "NtReadFile NtWriteFile NtClose NtReadFile NtWriteFile NtQueryInformationFile"
)

SINGLE_TEXTS = [MEDIUM_TEXT]                      # 1 window
BATCH_32     = [MEDIUM_TEXT] * 32                 # typical mini-batch
BATCH_64     = [MEDIUM_TEXT] * 64
BATCH_128    = [MEDIUM_TEXT] * 128
BATCH_300    = [MEDIUM_TEXT] * 300                # one full 30-second execution

ALL_LENGTHS  = {"short": SHORT_TEXT, "medium": MEDIUM_TEXT, "long": LONG_TEXT}

N_WARMUP  = 20
N_MEASURE = 100


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_batch(model, texts, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    """Returns per-window latency in ms (total batch time / batch size)."""
    batch_size = len(texts)
    for _ in range(n_warmup):
        model.encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size)
        sync()

    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        model.encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=batch_size)
        sync()
        times.append((time.perf_counter() - t0) * 1000)  # ms total

    per_win = np.array(times) / batch_size
    return {
        "batch_size":  batch_size,
        "total_ms":    {"mean": float(np.mean(times)),   "std": float(np.std(times)),
                        "p50":  float(np.percentile(times, 50)),
                        "p95":  float(np.percentile(times, 95))},
        "per_win_ms":  {"mean": float(np.mean(per_win)), "std": float(np.std(per_win)),
                        "p95":  float(np.percentile(per_win, 95))},
    }


def benchmark_model(name, model_id):
    """Load model, run full benchmark, return result dict."""
    print(f"\n{'─'*60}")
    print(f"  Loading {name} ({model_id}) ...")
    model = SentenceTransformer(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    emb_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {emb_dim}  |  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  |  "
              f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    result = {"model": name, "model_id": model_id, "emb_dim": emb_dim, "device": device}

    # Per text-length single-window timing
    print(f"  Benchmarking single-window across text lengths ...")
    result["single_window"] = {}
    for length_name, text in ALL_LENGTHS.items():
        r = time_batch(model, [text])
        result["single_window"][length_name] = r["per_win_ms"]
        mean_ms = r["per_win_ms"]["mean"]
        gate_ok = "✅" if mean_ms < GATE_MS else "✗ "
        print(f"    {length_name:<8} text: {mean_ms:7.2f} ms/window  {gate_ok}")

    # Batch throughput
    print(f"  Benchmarking batches (medium text) ...")
    result["batched"] = {}
    for texts, label in [(BATCH_32, "batch=32"), (BATCH_64, "batch=64"),
                         (BATCH_128, "batch=128"), (BATCH_300, "batch=300 (full exec)")]:
        r = time_batch(model, texts, n_warmup=10, n_measure=50)
        result["batched"][label] = r
        per_win = r["per_win_ms"]["mean"]
        total   = r["total_ms"]["mean"]
        gate_ok = "✅" if per_win < GATE_MS else "  "
        print(f"    {label:<24}: {per_win:7.3f} ms/window  (total {total:.1f} ms)  {gate_ok}")

    if device == "cuda":
        result["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1e9

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

device_str = "CUDA: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print("=" * 60)
print(f"D1 — mpnet Latency Benchmark")
print(f"Device: {device_str}")
print(f"Real-time gate: < {GATE_MS} ms per 0.1-s window")
print("=" * 60)

if not HAS_ST:
    print("ERROR: sentence_transformers not installed. Exiting.")
    exit(1)

all_results = []
for name, model_id in [
    ("MiniLM",  "sentence-transformers/all-MiniLM-L6-v2"),
    ("mpnet",   "sentence-transformers/all-mpnet-base-v2"),
]:
    r = benchmark_model(name, model_id)
    all_results.append(r)

# ── Summary report ─────────────────────────────────────────────────────────────

lines = []
def log(s=""): print(s, flush=True); lines.append(s)

log()
log("=" * 70)
log("D1 SUMMARY — Latency Gate Check (< 100 ms per window)")
log("=" * 70)
log(f"{'Model':<10}  {'Text length':<10}  {'Single-win ms':>14}  {'Batch=32 ms/win':>16}  {'Batch=300 ms/win':>17}  {'Gate OK?':>10}")
log("─" * 80)

for r in all_results:
    name = r["model"]
    for length_name in ["short", "medium", "long"]:
        sw   = r["single_window"][length_name]["mean"]
        b32  = r["batched"]["batch=32"]["per_win_ms"]["mean"]
        b300 = r["batched"]["batch=300 (full exec)"]["per_win_ms"]["mean"]
        gate = "✅ PASS" if sw < GATE_MS else "✗  FAIL"
        log(f"  {name:<8}  {length_name:<10}  {sw:>14.2f}  {b32:>16.3f}  {b300:>17.3f}  {gate:>10}")
    log()

log()
log("── Decision ──")
for r in all_results:
    sw_med = r["single_window"]["medium"]["mean"]
    b32_med = r["batched"]["batch=32"]["per_win_ms"]["mean"]
    b300_med = r["batched"]["batch=300 (full exec)"]["per_win_ms"]["mean"]
    name = r["model"]
    if sw_med < GATE_MS:
        log(f"  {name}: PASSES real-time gate at {sw_med:.2f} ms/window (medium text). "
            f"Batched=300: {b300_med:.3f} ms/win.")
        if name == "mpnet":
            log(f"  → mpnet is RE-OPENED as embedder candidate. Run D2 with mpnet embeddings.")
    else:
        log(f"  {name}: FAILS real-time gate at {sw_med:.2f} ms/window (medium text). "
            f"Batched=300: {b300_med:.3f} ms/win.")
        if name == "mpnet":
            log(f"  → mpnet remains rejected for real-time; evaluate only for offline/batched use.")

summary = "\n".join(lines)
(OUTPUT_DIR / "diag_d1_mpnet_latency.txt").write_text(summary)

with open(OUTPUT_DIR / "diag_d1_mpnet_latency.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved: diag_d1_mpnet_latency.txt / .json")
print("D1 done.")
