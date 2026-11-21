#!/bin/bash
#SBATCH --job-name=ts_tcc_ranonly
#SBATCH --account=def-marwa
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/ts_tcc_ranonly_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/ts_tcc_ranonly_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== TS-TCC Ransomware-Distance Head + Two-Sided (HANDOFF item 56, Tasks B1/B2/C) ==="
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
echo ""

python -u ~/ransmap/ts_tcc_ranonly.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/ts_tcc_ranonly_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/ts_tcc_ranonly_results.json 2>/dev/null
