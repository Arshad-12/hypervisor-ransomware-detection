#!/bin/bash
#SBATCH --job-name=ransmap_latency
#SBATCH --account=def-marwa
#SBATCH --time=0:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/latency_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/latency_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP end-to-end latency benchmark ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "GPU mem: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo ""

python -u ~/ransmap/latency_benchmark.py

echo ""
echo "End: $(date)"
