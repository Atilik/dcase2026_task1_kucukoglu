#!/bin/bash
#SBATCH --job-name=comb_sum
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/combined_summary_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/combined_summary_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

set -euo pipefail
PROJ_DIR=/scratch/$USER/repos/dcase2026_task1_baseline
cd "$PROJ_DIR"

./sing <<'EOF'
echo "=== Just running summarize for combined ==="
CONFIG_FILE=config_combined.yaml python -u summarize_results.py
echo "Done!"
EOF
