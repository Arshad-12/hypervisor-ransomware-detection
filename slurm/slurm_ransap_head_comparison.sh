#!/bin/bash
#SBATCH --job-name=ransap_heads
#SBATCH --account=def-marwa
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/ransap_head_comparison_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/ransap_head_comparison_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSAP Head Comparison (HANDOFF item 61) ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"
echo ""

RANSMAP=/scratch/aabdulj4/RanSMAP/features
RANSAP=/scratch/aabdulj4/RanSAP/features
for f in \
    "$RANSMAP/metadata.csv" \
    "$RANSMAP/text_labels.npy" \
    "$RANSMAP/text_sequences.npy" \
    "$RANSMAP/classifier_2way.pt" \
    "$RANSMAP/fewshot_split_cnn.npz" \
    "$RANSAP/ransap_feat5.npy" \
    "$RANSAP/ransap_text_seqs.npy" \
    "$RANSAP/ransap_labels.npy"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done
echo "All prerequisites found."
echo ""

python -u ~/ransmap/ransap_head_comparison.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/ransap_head_comparison_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/ransap_head_comparison_results.json 2>/dev/null
