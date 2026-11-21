#!/bin/bash
#SBATCH --job-name=comp_ablation
#SBATCH --account=def-marwa
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/component_ablation_final_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/component_ablation_final_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== Component Ablation, Final Pipeline (HANDOFF item 58, Task A) ==="
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
    "$FEAT/vae_nonran_window_scores.npy" \
    "$FEAT/lstm_scores_all.npy"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done
echo "All prerequisites found."
echo ""

python -u ~/ransmap/component_ablation_final.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/component_ablation_final_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/component_ablation_final_results.json 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/ts_tcc_unsup_window_scores.npy 2>/dev/null
