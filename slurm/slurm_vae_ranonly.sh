#!/bin/bash
#SBATCH --job-name=vae_ranonly
#SBATCH --account=def-marwa
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/vae_ranonly_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/vae_ranonly_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== VAE Retrained on RANSOMWARE ONLY (inverse) + Likelihood Ratio (HANDOFF item 55, Task B+C) ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"
echo ""

FEAT=/scratch/aabdulj4/RanSMAP/features
for f in \
    "$FEAT/metadata.csv" \
    "$FEAT/text_labels.npy" \
    "$FEAT/text_sequences.npy" \
    "$FEAT/classifier_2way.pt" \
    "$FEAT/fewshot_split_cnn.npz"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done
echo "All required prerequisites found."
if [ ! -f "$FEAT/vae_nonran_window_scores.npy" ]; then
    echo "WARNING: vae_nonran_window_scores.npy not found — Task C (likelihood ratio)"
    echo "will be skipped, but Task B (ransomware-only VAE) will still run."
fi
echo "NOTE: does NOT overwrite vae_fixed_* or vae_nonran_* — writes new vae_ranonly_* artifacts."
echo ""

python -u ~/ransmap/vae_ranonly.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/vae_ranonly_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/vae_ranonly_results.json 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/vae_ranonly_model.pt 2>/dev/null
