#!/bin/bash
#SBATCH --job-name=diag_score_dist
#SBATCH --account=def-marwa
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/aabdulj4/logs/diag_score_dist_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/diag_score_dist_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== VAE Score-Distribution Diagnostic (HANDOFF item 55, Task A) ==="
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "NOTE: CPU-only, no GPU needed -- pure post-hoc analysis of already-saved scores."
echo ""

FEAT=/scratch/aabdulj4/RanSMAP/features
for f in \
    "$FEAT/text_labels.npy" \
    "$FEAT/fewshot_split_cnn.npz" \
    "$FEAT/vae_fixed_window_scores.npy" \
    "$FEAT/vae_fixed_threshold.npy" \
    "$FEAT/vae_nonran_window_scores.npy" \
    "$FEAT/vae_nonran_threshold.npy"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done
echo "All prerequisites found."
echo ""

python -u ~/ransmap/diagnostic_score_dist.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/diagnostic_score_dist_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/diagnostic_score_dist_results.json 2>/dev/null
