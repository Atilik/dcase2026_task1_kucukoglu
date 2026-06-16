#!/bin/bash
#SBATCH --job-name=clsw_cw
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/classw_cw_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/classw_cw_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

set -euo pipefail
PROJ_DIR=/scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p "$PROJ_DIR/logs"
cd "$PROJ_DIR"

./sing <<'EOF'
echo "=== Class-weighted + confidence-weighted ==="
python -u main.py --config config_classw_cw.yaml
echo "Done!"
EOF
