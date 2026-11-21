#!/bin/bash
#SBATCH --job-name=ransmap_d3
#SBATCH --account=def-marwa
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/d3_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/d3_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP D3+D4+D5 — feat_23 Discriminative/Distance Anomaly Head ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo ""

python -u ~/ransmap/diag_d3_feat23_head.py

echo ""
echo "End: $(date)"
