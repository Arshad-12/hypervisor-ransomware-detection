#!/bin/bash
#SBATCH --job-name=ransmap_vae
#SBATCH --account=def-marwa
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/vae_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/vae_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP VAE Training (Full-Scale, 16c) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"

python -u ~/ransmap/vae_training.py

echo "End: $(date)"
