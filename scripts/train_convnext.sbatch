#!/bin/bash
#SBATCH --job-name=cnx_bl
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/convnext_train_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/convnext_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== ConvNeXt baseline ==="
python -u main.py --config config_convnext.yaml

echo "=== ConvNeXt + mixup 0.2 ==="
python -u main.py --config config_convnext_mix02.yaml
EOF
