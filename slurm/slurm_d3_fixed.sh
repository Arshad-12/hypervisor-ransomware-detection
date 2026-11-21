#!/bin/bash
#SBATCH --job-name=ransmap_d3fixed
#SBATCH --account=def-marwa
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/d3fixed_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/d3fixed_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP D3-FIXED: VAE retrain + D3 heads + full fusion eval ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "GPU mem: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo ""

python -u ~/ransmap/d3_fixed.py

echo ""
echo "End: $(date)"
