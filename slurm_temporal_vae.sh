#!/bin/bash
#SBATCH --job-name=ransmap_tvae
#SBATCH --account=def-marwa
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/temporal_vae_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/temporal_vae_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP Temporal VAE (LSTM, Full-Scale) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"

python -u ~/ransmap/temporal_vae_training.py

echo "End: $(date)"
