#!/bin/bash
#SBATCH --job-name=comb_s3
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/combined_sw03_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/combined_sw03_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

set -euo pipefail
PROJ_DIR=/scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p "$PROJ_DIR/logs"
cd "$PROJ_DIR"

./sing <<'EOF'
echo "=== Exp C: Combined + source weight BSD35k=0.3 ==="
python -u main.py --config config_combined_sw03.yaml
echo "Done!"
EOF
