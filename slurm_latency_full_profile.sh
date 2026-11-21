#!/bin/bash
#SBATCH --job-name=latency_full
#SBATCH --account=def-marwa
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/latency_full_profile_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/latency_full_profile_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== Complete Latency Profile: 4 stages, mean/median/P95 (HANDOFF item 66) ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"
echo ""

FEAT=/scratch/aabdulj4/RanSMAP/features
for f in \
    "$FEAT/metadata.csv" \
    "$FEAT/classifier_2way.pt"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    fi
done
echo "All prerequisites found."
echo ""
echo "NOTE: Stage 1 (feature extraction) reads raw per-execution CSVs from each row's"
echo "exec_dir in metadata.csv. If those raw CSVs are not present on this node (only the"
echo "pre-extracted feat_23 .npy files are guaranteed to exist under \$FEAT), Stage 1 will"
echo "report fewer than 100 successful timings and print a warning rather than fail --"
echo "check the WARNING line in the output below before trusting Stage 1's numbers."
echo ""

python -u ~/ransmap/latency_full_profile.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/RanSMAP/features/latency_full_profile_results.txt 2>/dev/null
