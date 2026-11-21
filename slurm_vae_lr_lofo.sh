#!/bin/bash
#SBATCH --job-name=vae_lr_lofo
#SBATCH --account=def-marwa
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/vae_lr_lofo_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/vae_lr_lofo_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== VAE Likelihood-Ratio LOFO (HANDOFF item 56, Task A) ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"
echo ""

FEAT=/scratch/aabdulj4/RanSMAP/features
for f in \
    "$FEAT/metadata.csv" \
    "$FEAT/text_labels.npy" \
    "$FEAT/text_sequences.npy" \
    "$FEAT/fewshot_split_cnn.npz"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done
echo "All prerequisites found."
echo ""

python -u ~/ransmap/vae_lr_lofo.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/vae_lr_lofo_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/vae_lr_lofo_results.json 2>/dev/null
