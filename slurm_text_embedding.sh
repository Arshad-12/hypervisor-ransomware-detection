#!/bin/bash
#SBATCH --job-name=ransmap_embed
#SBATCH --account=def-marwa
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/aabdulj4/logs/embed_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/embed_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11

source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP Text Embedding Job ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

python ~/ransmap/text_embedding.py

echo "End: $(date)"
