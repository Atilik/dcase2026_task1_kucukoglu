#!/bin/bash
#SBATCH --job-name=beats_exp1a
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/beats_exp1a_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/beats_exp1a_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

set -euo pipefail

PROJ_DIR=/scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p "$PROJ_DIR/logs"

cd "$PROJ_DIR"

./sing <<'EOF'
echo ""
echo "========================================"
echo "  DCASE 2026 Task 1 - BEATs Experiment"
echo "  Config: config_beats.yaml"
echo "========================================"
echo ""

python -u main.py --config config_beats.yaml

echo ""
echo "Done! Check model_output_beats/ for results."
EOF
