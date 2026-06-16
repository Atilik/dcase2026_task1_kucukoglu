#!/bin/bash
#SBATCH --job-name=cw_bin
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/cw_binary_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/cw_binary_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

set -euo pipefail
PROJ_DIR=/scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p "$PROJ_DIR/logs"
cd "$PROJ_DIR"

./sing <<'EOF'
echo "=== CW binary (conf>=3 → 1.0, else 0.3) ==="
python -u main.py --config config_cw_binary.yaml
echo "Done!"
EOF
