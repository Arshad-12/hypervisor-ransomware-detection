#!/bin/bash
#SBATCH --job-name=ransmap_fusion
#SBATCH --account=def-marwa
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/fusion_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/fusion_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP Fusion v2 (Full-Scale, 16d) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"

python -u ~/ransmap/fusion_v2.py

echo "End: $(date)"
