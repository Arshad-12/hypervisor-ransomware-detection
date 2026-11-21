#!/bin/bash
#SBATCH --job-name=ransmap_reextract
#SBATCH --account=def-marwa
#SBATCH --time=4:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/aabdulj4/logs/reextract_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/reextract_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSMAP feat_23 Re-extraction (count_page_access bug fix) ==="
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

python -u ~/ransmap/reextract_feat23.py

echo ""
echo "End: $(date)"
