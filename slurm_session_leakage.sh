#!/bin/bash
#SBATCH --job-name=session_leakage
#SBATCH --account=def-marwa
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/session_leakage_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/session_leakage_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== Session-Leakage Diagnostic (HANDOFF item 59, professor comment) ==="
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
    "$FEAT/fewshot_split_cnn.npz" \
    "$FEAT/lstm_scores_all.npy"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done
echo "All prerequisites found."
echo ""

python -u ~/ransmap/session_leakage_diagnostic.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/session_leakage_diagnostic_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/session_leakage_diagnostic_results.json 2>/dev/null
