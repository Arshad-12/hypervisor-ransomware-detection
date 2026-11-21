#!/bin/bash
#SBATCH --job-name=ransap_extract
#SBATCH --account=def-marwa
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/ransap_extract_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/ransap_extract_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSAP Feature Extraction + MiniLM Embedding ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "GPU mem: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"
echo ""

# Check RanSAP data exists
if [ ! -d /scratch/aabdulj4/RanSAP/RanSAP-main ]; then
    echo "ERROR: /scratch/aabdulj4/RanSAP/RanSAP-main does not exist."
    echo "Run: cd /scratch/aabdulj4/RanSAP && unzip ransap_main.zip"
    exit 1
fi

echo "RanSAP dataset contents:"
ls /scratch/aabdulj4/RanSAP/RanSAP-main/dataset/ 2>/dev/null || ls /scratch/aabdulj4/RanSAP/RanSAP-main/
echo ""
echo "Searching for ata_write.csv files..."
find /scratch/aabdulj4/RanSAP/RanSAP-main -name "ata_write.csv" | head -5
echo "Total ata_write.csv files: $(find /scratch/aabdulj4/RanSAP/RanSAP-main -name 'ata_write.csv' 2>/dev/null | wc -l)"
echo ""

python -u ~/ransmap/ransap_extract.py

echo ""
echo "End: $(date)"

echo ""
echo "Output files:"
ls -lh /scratch/aabdulj4/RanSAP/features/ 2>/dev/null || echo "(features dir not found)"
