#!/bin/bash
#SBATCH --job-name=cerber_diag
#SBATCH --account=def-marwa
#SBATCH --time=1:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/cerber_diag_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/cerber_diag_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== Cerber Diagnosis ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"
echo ""

python -u ~/ransmap/cerber_diagnosis.py

echo ""
echo "End: $(date)"
echo ""
echo "Outputs:"
ls -lh /scratch/aabdulj4/logs/cerber_entropy_plot.png 2>/dev/null
ls -lh /scratch/aabdulj4/logs/cerber_diagnosis.log 2>/dev/null
