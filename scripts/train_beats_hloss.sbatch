#!/bin/bash
#SBATCH --job-name=beats_hloss
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/beats_hloss_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/beats_hloss_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

set -euo pipefail

PROJ_DIR=/scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p "$PROJ_DIR/logs"

cd "$PROJ_DIR"

./sing <<'EOF'
echo ""
echo "=============================================="
echo "  DCASE 2026 Task 1 - BEATs + Hierarchical Loss"
echo "  Config: config_beats_hloss.yaml"
echo "=============================================="
echo ""

python -u main.py --config config_beats_hloss.yaml

echo ""
echo "Done! Check model_output_beats_hloss/ for results."
EOF
