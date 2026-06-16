#!/bin/bash
#SBATCH --job-name=cnx_ext
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/convnext_extract_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/convnext_extract_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Installing audioset_convnext_inf ==="
pip install git+https://github.com/topel/audioset-convnext-inf@pip-install 2>&1 | tail -3

echo "=== Extracting ConvNeXt embeddings ==="
python -u extract_convnext_embeddings.py
EOF
