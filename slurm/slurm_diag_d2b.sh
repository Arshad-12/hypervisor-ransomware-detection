#!/bin/bash
#SBATCH --job-name=ransmap_d2b
#SBATCH --account=def-marwa
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/d2b_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/d2b_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

pip install sentence-transformers --quiet --break-system-packages 2>/dev/null || true

echo "=== RanSMAP D2b — mpnet Re-embedding + Oracle ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo ""

python -u ~/ransmap/diag_d2b_mpnet_embed.py

echo ""
echo "End: $(date)"
