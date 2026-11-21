#!/bin/bash
#SBATCH --job-name=ransmap_binfus
#SBATCH --account=def-marwa
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/binfus_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/binfus_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP binary_fusion: 2-way CNN + binary fusion ablation ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "GPU mem: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo ""

python -u ~/ransmap/binary_fusion.py

echo ""
echo "End: $(date)"
