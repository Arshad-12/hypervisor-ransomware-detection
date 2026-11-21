#!/bin/bash
#SBATCH --job-name=ransmap_mpfus
#SBATCH --account=def-marwa
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/mpfus_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/mpfus_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP mpnet full fusion (fixed streaming, no training) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo ""

python -u ~/ransmap/mpnet_full_fusion.py

echo ""
echo "End: $(date)"
