#!/bin/bash
#SBATCH --job-name=concat_w100
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/concat_w100_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/concat_w100_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

set -euo pipefail
PROJ_DIR=/scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p "$PROJ_DIR/logs"
cd "$PROJ_DIR"

./sing <<'EOF'
echo "=== Concat + hloss weight=1.0 ==="
python -u main.py --config config_concat_hloss_w100.yaml
echo "Done!"
EOF
