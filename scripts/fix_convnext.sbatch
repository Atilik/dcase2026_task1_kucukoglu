#!/bin/bash
#SBATCH --job-name=cnx_fix
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/cnx_fix_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/cnx_fix_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Step 1: Re-extract ConvNeXt (fix 169 short audio files) ==="
python -u extract_convnext_embeddings.py

echo "=== Step 2: Rebuild data_convnext ==="
CONFIG_FILE=config_convnext.yaml python -u build_dataset.py

echo "=== Step 3: Train ConvNeXt baseline ==="
CONFIG_FILE=config_convnext.yaml python -u main.py --config config_convnext.yaml

echo "=== Step 4: Train ConvNeXt + mixup 0.2 ==="
CONFIG_FILE=config_convnext_mix02.yaml python -u main.py --config config_convnext_mix02.yaml
EOF
