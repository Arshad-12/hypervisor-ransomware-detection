#!/bin/bash
# ransap_download.sh — Download RanSAP dataset to scratch on Alliance cluster
# Run this on a LOGIN NODE (internet access). Do NOT submit as a SLURM job.
#
# RanSAP: Manabu Hirano (2022), CC BY-ND 4.0
# Kaggle: https://www.kaggle.com/datasets/hiranomanabu/ransap-2022-ransomware-behavioral-features
# GitHub: https://github.com/manabu-hirano/RanSAP
#
# Cite: Hirano, M. & Kobayashi, R. (2022). RanSAP: An open dataset of
#       ransomware storage access patterns for training ML-based detectors.
#       Forensic Science International: Digital Investigation, 40, 301314.

set -e
DEST=/scratch/aabdulj4/RanSAP
mkdir -p "$DEST"
cd "$DEST"

echo "=== RanSAP Download ==="
echo "Destination: $DEST"
echo ""

# ── Option 1: Kaggle API (recommended — downloads everything at once) ──────────
# Requires: pip install kaggle + ~/.kaggle/kaggle.json with your API key
# Get your API key at: https://www.kaggle.com/settings/account → Create New Token
#
# module load python/3.11
# source ~/ransmap/ransmap_env/bin/activate
# pip install kaggle --quiet
# kaggle datasets download hiranomanabu/ransap-2022-ransomware-behavioral-features \
#     --unzip --path "$DEST"

# ── Option 2: Direct wget from GitHub releases ─────────────────────────────────
# Check https://github.com/manabu-hirano/RanSAP/releases for current URLs.
# If the data is split into multiple zips, download each:
#
echo "Attempting download from GitHub..."
# Try the main repository zip (adjust URL if release assets differ)
wget -c "https://github.com/manabu-hirano/RanSAP/archive/refs/heads/main.zip" \
     -O ransap_main.zip 2>&1 | tail -5 || true

if [ -f ransap_main.zip ]; then
    echo "Extracting..."
    unzip -q ransap_main.zip
    echo "Done. Contents:"
    ls -la "$DEST"/
else
    echo ""
    echo "Automatic download failed. Please download manually:"
    echo "  1. Visit https://www.kaggle.com/datasets/hiranomanabu/ransap-2022-ransomware-behavioral-features"
    echo "  2. Download and extract to: $DEST"
    echo "  3. Expected structure after extraction:"
    echo "       $DEST/"
    echo "         ransomware/"
    echo "           Cerber/  WannaCry/  Sodinokibi/  ..."
    echo "         benign/"
    echo "         bitlocker/"
    echo "     OR whatever structure the dataset uses."
    echo ""
    echo "Then run: ls -la $DEST/ and tell Sonnet the directory structure."
fi

echo ""
echo "After download, check structure with:"
echo "  find $DEST -name 'ata_write.csv' | head -5"
echo "  find $DEST -name 'ata_write.csv' | wc -l"
