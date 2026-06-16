#!/bin/bash
#SBATCH --job-name=ext_weak
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/ext_weak_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/ext_weak_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Strategy A: External Weak Classes Only ==="
echo "Config: config_ext_weak.yaml"
echo "Only sp-c (213) and ss-i (341) external samples"
echo "Start: $(date)"
CONFIG_FILE=config_ext_weak.yaml python -u main.py --config config_ext_weak.yaml
echo "End: $(date)"
EOF
