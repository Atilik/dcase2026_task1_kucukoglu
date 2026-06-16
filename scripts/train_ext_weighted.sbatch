#!/bin/bash
#SBATCH --job-name=ext_wt
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/ext_weighted_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/ext_weighted_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Strategy A: External Weighted ==="
echo "Config: config_ext_weighted.yaml"
echo "All external, BSD10k=1.0, ESC-50=0.3, FSD50K=0.3"
echo "Start: $(date)"
CONFIG_FILE=config_ext_weighted.yaml python -u main.py --config config_ext_weighted.yaml
echo "End: $(date)"
EOF
