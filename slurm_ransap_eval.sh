#!/bin/bash
#SBATCH --job-name=ransap_eval
#SBATCH --account=def-marwa
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/aabdulj4/logs/ransap_eval_%j.out
#SBATCH --error=/scratch/aabdulj4/logs/ransap_eval_%j.err

mkdir -p /scratch/aabdulj4/logs

module load python/3.11
source ~/ransmap/ransmap_env/bin/activate

echo "=== RanSAP Cross-Dataset Validation ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "GPU mem: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)"
echo "Start:   $(date)"
echo ""

# Check prerequisites
for f in /scratch/aabdulj4/RanSAP/features/ransap_feat5.npy \
          /scratch/aabdulj4/RanSAP/features/ransap_text_seqs.npy \
          /scratch/aabdulj4/RanSAP/features/ransap_labels.npy \
          /scratch/aabdulj4/RanSMAP/features/metadata.csv \
          /scratch/aabdulj4/RanSMAP/features/text_labels.npy \
          /scratch/aabdulj4/RanSMAP/features/fewshot_split_cnn.npz \
          /scratch/aabdulj4/RanSMAP/features/classifier_2way.pt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing required file: $f"
        exit 1
    fi
done
echo "All prerequisites found."
echo ""

python -u ~/ransmap/ransap_eval.py

echo ""
echo "End: $(date)"
