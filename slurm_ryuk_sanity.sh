#!/bin/bash
#SBATCH --job-name=ransmap_ryuksanity
#SBATCH --account=def-marwa
#SBATCH --time=1:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/ryuksanity_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/ryuksanity_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP Ryuk LOFO sanity check ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo ""

python -u ~/ransmap/ryuk_lofo_sanity.py

echo ""
echo "End: $(date)"
