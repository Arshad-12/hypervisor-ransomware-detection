#!/bin/bash
#SBATCH --job-name=flat_vae_fusion
#SBATCH --account=def-marwa
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/flat_vae_fusion_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/flat_vae_fusion_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== Flat VAE Binary Fusion ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"
echo ""

FEAT=/scratch/aabdulj4/RanSMAP/features
for f in \
    "$FEAT/vae_fixed_model.pt" \
    "$FEAT/vae_fixed_window_scores.npy" \
    "$FEAT/vae_fixed_threshold.npy" \
    "$FEAT/classifier_2way.pt" \
    "$FEAT/text_sequences.npy" \
    "$FEAT/text_labels.npy" \
    "$FEAT/metadata.csv" \
    "$FEAT/fewshot_split_cnn.npz" \
    "$FEAT/lstm_scores_all.npy" \
    "$FEAT/lstm_threshold.npy"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done
echo "All prerequisites found."
echo ""

python -u ~/ransmap/flat_vae_binary_fusion.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/flat_vae_binary_fusion_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/flat_vae_binary_fusion_results.json 2>/dev/null
