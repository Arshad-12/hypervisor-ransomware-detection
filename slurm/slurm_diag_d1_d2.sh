#!/bin/bash
#SBATCH --job-name=ransmap_diag
#SBATCH --account=def-marwa
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/diag_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/diag_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

# Install sentence-transformers if not present
pip install sentence-transformers --quiet --break-system-packages 2>/dev/null || true

echo "=== RanSMAP Diagnostics D1 + D2 ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"

echo ""
echo "--- D1: mpnet Latency Benchmark ---"
python -u ~/ransmap/diag_d1_mpnet_latency.py

echo ""
echo "--- D2: Supervised Separability Oracle ---"
python -u ~/ransmap/diag_d2_separability.py

echo ""
echo "End: $(date)"
