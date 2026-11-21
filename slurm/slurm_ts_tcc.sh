#!/bin/bash
#SBATCH --job-name=ts_tcc_detection
#SBATCH --account=def-marwa
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/ts_tcc_detection_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/ts_tcc_detection_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== TS-TCC Self-Supervised Detector (HANDOFF item 53) ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"
echo ""

FEAT=/scratch/aabdulj4/RanSMAP/features
for f in \
    "$FEAT/metadata.csv" \
    "$FEAT/text_labels.npy" \
    "$FEAT/fewshot_split_cnn.npz"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done
echo "All prerequisites found."
echo "NOTE: this job reads raw feat_23 (300,23) sequences directly from each"
echo "execution's .filepath in metadata.csv (same as d3_fixed.py) — no"
echo "separate feat_23 array file is required."
echo ""

python -u ~/ransmap/ts_tcc_detection.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/ts_tcc_detection_results.txt 2>/dev/null
ls -lh /scratch/aabdulj4/RanSMAP/features/ts_tcc_detection_results.json 2>/dev/null
