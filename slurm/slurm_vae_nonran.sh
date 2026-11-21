#!/bin/bash
#SBATCH --job-name=vae_nonran
#SBATCH --account=def-marwa
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/vae_nonran_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/vae_nonran_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== VAE Retrained on NON-RANSOMWARE ONLY (HANDOFF item 54) ==="
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
echo "All prerequisites found."
echo "NOTE: this job does NOT overwrite vae_fixed_* (the all-classes VAE) —"
echo "it writes new vae_nonran_* artifacts alongside it."
echo ""

python -u ~/ransmap/vae_nonran_only.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/vae_nonran_only_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/vae_nonran_only_results.json 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/vae_nonran_model.pt 2>/dev/null
